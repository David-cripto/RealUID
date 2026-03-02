import argparse
import random
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception:
    print("Not npu case")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


import torch.nn as nn
import copy
import os
import gc
import pickle
import json

from torchvision import datasets, transforms
from tqdm import trange
from src.train_utils import ema, infiniteloop, TensorBoardWriter, WandBWriter, CustomImageDataset
import numpy as np


from src.models import UNetModelWrapperWithHead
from src.generate import gen_function, generate_and_save_samples
from src.losses import dist_loss, GANloss, general_dist_loss
from src.eval import eval_cifar_fid, eval_fid

  

parser = argparse.ArgumentParser(description="alpha beta parser")

#loss 
parser.add_argument('--alpha', type=float, default=1.00, help='RealUID alpha')
parser.add_argument('--beta', type=float, default=1.00, help='RealUID beta')
parser.add_argument('--adv_step', type=int, default=6)

parser.add_argument('--gamma', type=float, default=None, help='if gamma is given the general loss is used. It is equivalent to the original in case of standard parameterization and alpha = gamma')
parser.add_argument('--parameterization', default='standard',  choices=['standard', 'beta'], help = 'alternative parameterizations for general loss')

parser.add_argument('--gen_coef', type=float, default=None, help='gen_coef') 
parser.add_argument('--disc_coef', type=float, default=None, help='disc_coef')
parser.add_argument('--with_gan_loss', action='store_true', help='with_gan_loss')



#paths
parser.add_argument('--output_dir', type=str)
parser.add_argument('--exp_name', type=str, default=None, help='name of the subfolder in the output dir') 

parser.add_argument('--teacher_ckpt_path', type=str, help = 'teacher model checkpoint')
parser.add_argument('--teacher_ema', type=float, default=0.999, help = 'teacher ema to select the net from checkpoint')


parser.add_argument('--resume_ckpt_path', type=str, default=None, help = 'continue training from checkpoint if chosen')

parser.add_argument('--finetune_ckpt_path', type=str, default=None, help = 'fine-tune the checkpoint if chosen')
parser.add_argument('--finetune_ema', type=float, default=None, help = 'fine-tune ema to select the net from checkpoint')


#dataset
parser.add_argument('--dataset', default='cifar10',  choices=['cifar10', 'celeba'])
parser.add_argument('--cond', default='uncond',  choices=['cond', 'uncond'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--celeba_dir', type=str, default=None, help='folder with celeba dataset images if chosen')
parser.add_argument('--num_classes', type=int, default=10)


#logger
parser.add_argument('--logger_name', default=None,  choices=['tensorboard', 'wandb'])
parser.add_argument('--wandb_user', type=str, default=None)
parser.add_argument('--wandb_api_key', type=str, default=None)


#optimizer 
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate') #1e-5 for cifar finetuning,
parser.add_argument('--total_steps', type=int, default=800_000) #1_600_000 for CelebA
parser.add_argument('--batch_size', type=int, default=256) #64 for CelebA
parser.add_argument('--warmup', type=int, default=500) #0 for fine-tuning
parser.add_argument('--ema_decays', nargs='+', type=float, default = [0.999, 0.9996, 0.9999], help='list of ema decays to track')
parser.add_argument('--num_workers', type=int, default=1) 
parser.add_argument('--parallel', action='store_true', help='parallel mode')
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)

#eval
parser.add_argument('--fid_step', type=int, default=6000, help='eval generator fid once per the given number of iterations') 
parser.add_argument('--save_image_step', type=int, default=6000, help='save generated images once per the given number of iterations') 
parser.add_argument('--num_save_image', type=int, default=100, help='number of samples to generate') 
parser.add_argument('--fid_counts', type=int, default=3, help='average fid across several attempts') 


parser.add_argument('--save_model_step', type=int, default=6000, help='save model checkpoint once per the given number of iterations') 
parser.add_argument('--num_gen', type=int, default=50_000, help = 'number of generated samples for fid evaluation, do not change') 
parser.add_argument('--fid_batch_size', type=int, default=100, help ='batch size for evaluating fid, it must be divisible by the num_classes and must divide num_gen if conditional') 



args = parser.parse_args()

alpha = args.alpha #0.85 - 1.00
beta = args.beta #0.85 - 1.00 
gamma = args.gamma #0.85 - 1.00 
parameterization = args.parameterization
adv_step = args.adv_step 


lr = args.lr
total_steps = args.total_steps
warmup =  max(1, args.warmup)
batch_size = args.batch_size
num_workers = args.num_workers
parallel = args.parallel
grad_clip = args.grad_clip
ema_decays = args.ema_decays


with_gan_loss = args.with_gan_loss
gen_coef =  args.gen_coef # 0.3 1.0 5.0 25.0  the best set is (5.0, 15.0)
disc_coef = args.disc_coef # 1.0 3.0 15.0 75.0 
resume_ckpt_path = args.resume_ckpt_path
finetune_ckpt_path = args.finetune_ckpt_path
COND = args.cond == 'cond' #conditional or unconditional
num_classes = args.num_classes

fid_batch_size = args.fid_batch_size
num_gen = args.num_gen
fid_step = args.fid_step
save_image_step = args.save_image_step
save_model_step = args.save_model_step
fid_counts = args.fid_counts
num_save_image = args.num_save_image
dataset_name = args.dataset


if COND:
    assert fid_batch_size % num_classes == 0 and num_gen % fid_batch_size == 0, \
        f"fid_batch_size ({fid_batch_size}) must be divisible by num_classes ({num_classes}) and divide num_gen ({num_gen}) if conditional"


#Set seed
seed = args.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)




logger_name = args.logger_name

if with_gan_loss:
    assert (gen_coef is not None or disc_coef is not None), "Initialize gan coefs!"
else:
    assert (gen_coef is None or disc_coef is None), "If gan coefs are not None add --with_gan_loss!"

exp_name = args.exp_name
if exp_name is None:
    exp_name = f'alpha{alpha}_beta{beta}' 

    if gamma is not None:
        exp_name += f'_gamma{gamma}_param_{parameterization}'
    if with_gan_loss:
        exp_name += f'_use_gan_gen_coef_{gen_coef}_disc_coef_{disc_coef}'

    if resume_ckpt_path is not None:
        exp_name = exp_name + "_continue"
    if finetune_ckpt_path is not None:
        exp_name = exp_name + "_finetune"   

    exp_name = f'{args.dataset}_{args.cond}_{exp_name}'

output_dir = args.output_dir


savedir = os.path.join(output_dir, exp_name)
os.makedirs(savedir, exist_ok=True)

args_dict = vars(args)  

# Save to file
with open(os.path.join(savedir, 'arguments.json'), 'w') as f:
    json.dump(args_dict, f, indent=4)
    
if logger_name is None:
    logger = None
else:   
    logger = TensorBoardWriter(savedir) if logger_name == "tensorboard" else WandBWriter(args, savedir, exp_name)


 
#LOAD REAL DATA    

if dataset_name == 'cifar10':
    trans = transforms.Compose(
                [  
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],  
                                std=[0.5, 0.5, 0.5])
                ]
            )


    dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=trans
            
        )
elif dataset_name == 'celeba':

    trans = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],  
                                        std=[0.5, 0.5, 0.5])
                ]
            )

    dataset = CustomImageDataset(args.celeba_dir, 202599, trans)

dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers = num_workers,
        drop_last=True,
    )
datalooper = infiniteloop(dataloader)


###Init allmodels

resolution = 32 if dataset_name == 'cifar10' else 64

net_model_for_dist = UNetModelWrapperWithHead(
        dim=(3, resolution, resolution),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=args.dropout,
        class_cond = COND,
        num_classes = num_classes
    ).to(device)  



u = copy.deepcopy(net_model_for_dist)  


# Load the model conditional

print("teacher path: ", args.teacher_ckpt_path)
teacher_checkpoint = torch.load(args.teacher_ckpt_path, map_location=device)
net_model_for_dist.load_state_dict(teacher_checkpoint["ema_models"][str(args.teacher_ema)], strict = False)
net_model_for_dist.eval()

generator = copy.deepcopy(net_model_for_dist)  
ema_gens = {str(d): copy.deepcopy(generator) for d in ema_decays}
for m in ema_gens.values():
    m.requires_grad_(False)

for param in net_model_for_dist.parameters():
    param.requires_grad = False   
 
#### Init optimizers




optim_gen = torch.optim.Adam(generator.parameters(),betas=(0.0, 0.999), lr=lr) 
optim_u = torch.optim.Adam(u.parameters(), betas=(0.0, 0.999), maximize = True, lr=lr) 


def warmup_lr(step):
    return min(step, warmup) / warmup
sched_gen = torch.optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=warmup_lr)
sched_u = torch.optim.lr_scheduler.LambdaLR(optim_u, lr_lambda=warmup_lr)


init_step = 0

if resume_ckpt_path is not None:
    checkpoint = torch.load(resume_ckpt_path, map_location=device)
    generator.load_state_dict(checkpoint["gen"], strict = False)

    for d in ema_decays:
        k = str(d)
        ema_gens[k].load_state_dict(checkpoint["ema_gens"][k], strict=False)
    
    
    
    u.load_state_dict(checkpoint["u"], strict = False)
    
    optim_gen.load_state_dict(checkpoint["optim_gen"]) 
    optim_u.load_state_dict(checkpoint["optim_u"])
    
    sched_gen.load_state_dict(checkpoint["sched"])
    sched_u.load_state_dict(checkpoint["sched"])
    init_step = checkpoint["step"]



if finetune_ckpt_path is not None:
    checkpoint = torch.load(finetune_ckpt_path, map_location=device)
    generator.load_state_dict(checkpoint["ema_gens"][str(args.finetune_ema)], strict = False)

    for d in ema_decays:
        k = str(d)
        ema_gens[k].load_state_dict(checkpoint["ema_gens"][str(args.finetune_ema)], strict=False)
    
    
    
    u.load_state_dict(teacher_checkpoint["ema_models"][str(args.teacher_ema)], strict = False)
    
    




fids = []
ema_fids = {str(d): [] for d in ema_decays}

with trange(init_step, total_steps, dynamic_ncols=True) as pbar:
        
        for step in pbar:
            gc.collect()
            x1_data, y = next(datalooper)
            x1_data = x1_data.to(device)
            
            if COND:
                y = y.to(device)
            else:
                y = None
                
            x0 = torch.randn_like(x1_data)
            t = torch.rand(x0.shape[0]).type_as(x0)
            z = torch.randn_like(x0)
            
            t_gan = 1.0 - 0.2 * torch.rand(x0.shape[0]).type_as(x0)   #the best set - 0.8 - 1.0         
    

            if step % adv_step == adv_step - 1:
                    u.eval()
                    optim_gen.zero_grad()
                    
                    x1_gen = gen_function(generator, z, y)

                    if gamma is None:
                        loss = dist_loss(u, net_model_for_dist, t, x0, x1_gen, x1_data, y, alpha, beta, generator_turn=True)
                    else:
                        loss = general_dist_loss(u, net_model_for_dist, t, x0, x1_gen, x1_data, y, alpha, beta, gamma, parameterization, generator_turn=True)

                    if with_gan_loss:
                        gan_loss_gen = GANloss(u, t_gan, x0, x1_gen, x1_data, y, generator_turn=True) 
                        loss =  loss + gen_coef * gan_loss_gen 
                        if logger is not None:
                            logger.add_scalar(step, "GAN loss gen", gan_loss_gen.detach().cpu().numpy())
                            logger.add_scalar(step, "loss gen", loss.detach().cpu().numpy())
                    else:
                        if logger is not None:
                            logger.add_scalar(step, "loss gen", loss.detach().cpu().numpy())
                            
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)  

                    optim_gen.step()
                    sched_gen.step()
                    
                   
                    u.train()
                    for d in ema_decays:
                        ema(generator, ema_gens[str(d)], d)

            else:
                    generator.eval()
                    optim_u.zero_grad()
                    x1_gen = gen_function(generator, z, y)
                    x1_gen = x1_gen.detach()

                    if gamma is None:
                        loss = dist_loss(u, net_model_for_dist, t, x0, x1_gen, x1_data, y, alpha , beta, generator_turn=False)
                    else:
                        loss = general_dist_loss(u, net_model_for_dist, t, x0, x1_gen, x1_data, y, alpha, beta, gamma, parameterization, generator_turn=False)

                    if with_gan_loss:
                        gan_loss = GANloss(u, t_gan, x0, x1_gen, x1_data, y, generator_turn=False) 
                        loss = loss + disc_coef * gan_loss 
                        if logger is not None:
                            logger.add_scalar(step, "GAN loss", gan_loss.detach().cpu().numpy())
                            logger.add_scalar(step, "loss", loss.detach().cpu().numpy())
                    else:
                        if logger is not None:
                            logger.add_scalar(step, "loss", loss.detach().cpu().numpy())
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(u.parameters(), grad_clip)  

                    optim_u.step()
                    sched_u.step()
                    generator.train()



            # sample and save the weights
            if save_image_step > 0 and step % save_image_step == 0:
                with torch.no_grad():
    
                    if COND:
                         y = torch.arange(num_save_image, device=device, dtype=int) % num_classes  
                    else:
                         y = None

                    generate_and_save_samples(generator, savedir, y = y, name ="normal", batch_size=num_save_image, step = step, logger = logger)

                    for d in ema_decays:
                        tag = str(d).replace(".", "p")  # Safe for filenames
                        generate_and_save_samples(ema_gens[str(d)], savedir, y = y, name =f"ema_{tag}", batch_size=num_save_image, step = step, logger = logger)


            if fid_step > 0 and step % fid_step == 0: 
                        
                        scores = []
                        for _ in range(fid_counts): 
                            if dataset_name == 'cifar10':
                                scores.append(eval_cifar_fid(generator, num_gen=num_gen, fid_batch_size = fid_batch_size))
                            elif dataset_name == 'celeba':

                                gen_dir = os.path.join(savedir, 'gen_celeba' )
                                scores.append(eval_fid(generator, gen_dir, args.celeba_dir, num_gen=num_gen, fid_batch_size = fid_batch_size))

                        fids.append(np.mean(scores))
                        
                        with open(os.path.join(savedir,  f"fids.pkl"), 'wb') as file:
                           pickle.dump(fids, file)
                            
                        if logger is not None:
                            logger.add_scalar(step, "fid", np.mean(scores))

                        

                        for d in ema_decays:
                            k = str(d)
                            scores = []
                            for _ in range(fid_counts):
                                if dataset_name == 'cifar10':
                                    scores.append(eval_cifar_fid(ema_gens[k], num_gen=num_gen, fid_batch_size = fid_batch_size))
                                elif dataset_name == 'celeba':
                                    gen_dir = os.path.join(savedir, 'gen_celeba' )
                                    scores.append(eval_fid(ema_gens[k], gen_dir, args.celeba_dir, num_gen=num_gen, fid_batch_size = fid_batch_size))
                            
                            mean_score = float(np.mean(scores))

                            ema_fids[k].append(mean_score)

                            if logger is not None:
                                tag = k.replace(".", "p")
                                logger.add_scalar(step, f"ema_{tag}_fid", mean_score)

                        # Save all EMA FID scores together
                            with open(os.path.join(savedir, "ema_fids.pkl"), "wb") as file:
                                pickle.dump(ema_fids, file)
            

            if save_model_step > 0 and step % save_model_step == 0:
                torch.save(
                            {
                                "gen": generator.state_dict(),
                                "u": u.state_dict(),
                                "ema_gens": {k: m.state_dict() for k, m in ema_gens.items()},  # New format for multiple EMA models
                                "sched": sched_gen.state_dict(),
                                "optim_gen": optim_gen.state_dict(),
                                "optim_u": optim_u.state_dict(),
                                "step": step,
                            },
                            os.path.join(savedir, f"{dataset_name}_{args.cond}_dist_step_{step}.pt"),
                )
