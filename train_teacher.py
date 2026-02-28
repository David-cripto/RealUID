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
from src.generate import generate_and_save_samples

from src.eval import eval_cifar_fid, eval_fid

from torchcfm.conditional_flow_matching import pad_t_like_x
import torch.nn.functional as F

  

parser = argparse.ArgumentParser(description="alpha beta parser")



#paths
parser.add_argument('--output_dir', type=str) 
parser.add_argument('--exp_name', type=str, default=None, help='name of the subfolder in the output dir') 
parser.add_argument('--resume_ckpt_path', type=str, default=None, help = 'continue training from checkpoint if chosen')


#dataset
parser.add_argument('--dataset',  choices=['cifar10', 'celeba'])
parser.add_argument('--cond', default='uncond',  choices=['cond', 'uncond'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--celeba_dir', type=str, default=None, help='folder with celeba dataset images if chosen') 
parser.add_argument('--num_classes', type=int, default=10)


#logger
parser.add_argument('--logger_name', default=None,  choices=['tensorboard', 'wandb'])
parser.add_argument('--wandb_user', type=str, default=None)
parser.add_argument('--wandb_api_key', type=str, default=None)

#optimizer 
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') #4e-5 for celeba
parser.add_argument('--total_steps', type=int, default=400_000)  
parser.add_argument('--batch_size', type=int, default=256) #64 for CelebA
parser.add_argument('--warmup', type=int, default=5000) 
parser.add_argument('--ema_decays', nargs='+', type=float, default = [0.999, 0.9996, 0.9999], help='list of ema decays to track')
parser.add_argument('--num_workers', type=int, default=1) 
parser.add_argument('--parallel', action='store_true', help='parallel mode')
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.1)
#eval
parser.add_argument('--fid_step', type=int, default=20_000, help='eval generator fid once per the given number of iterations') 
parser.add_argument('--save_image_step', type=int, default=5000, help='save generated images once per the given number of iterations') 
parser.add_argument('--num_save_image', type=int, default=100, help='number of samples to generate') 

parser.add_argument('--save_model_step', type=int, default=20_000, help='save model checkpoint once per the given number of iterations') 
parser.add_argument('--num_gen', type=int, default=50_000, help = 'number of generated samples for fid evaluation, do not change') 
parser.add_argument('--fid_batch_size', type=int, default=100, help ='batch size for evaluating fid, it must be divisible by the number of classes and must divide num_gen if conditional') 

args = parser.parse_args()



lr = args.lr
total_steps = args.total_steps
warmup = max(1, args.warmup)
batch_size = args.batch_size
num_workers = args.num_workers
parallel = args.parallel
grad_clip = args.grad_clip
ema_decays = args.ema_decays



resume_ckpt_path = args.resume_ckpt_path
COND = args.cond == 'cond' #conditional or unconditional
num_classes = args.num_classes


num_gen = args.num_gen
fid_step = args.fid_step
save_image_step = args.save_image_step
save_model_step = args.save_model_step
num_save_image = args.num_save_image
dataset_name = args.dataset
fid_batch_size = args.fid_batch_size

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
exp_name = args.exp_name


if exp_name is None:
    exp_name = 'FM'
    if resume_ckpt_path is not None:
        exp_name = exp_name + "_continue"
    exp_name = f'{args.dataset}_{args.cond}_{exp_name}'



output_dir = args.output_dir


savedir = os.path.join(output_dir, exp_name)
os.makedirs(savedir, exist_ok=True)

args_dict = vars(args)  

# Save arguments to file
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

u = UNetModelWrapperWithHead(
        dim=(3, resolution, resolution),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=args.dropout,
        class_cond = COND,
        num_classes = num_classes,
    ).to(device)  

 
ema_models = {str(d): copy.deepcopy(u) for d in ema_decays}
for m in ema_models.values():
    m.requires_grad_(False)
   
 
#### Init optimizers



 
optim = torch.optim.Adam(u.parameters(), betas=(0.9, 0.999), lr=lr) 


def warmup_lr(step):
    return min(step, warmup) / warmup
sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)


init_step = 0

if resume_ckpt_path is not None:
    checkpoint = torch.load(resume_ckpt_path, map_location=device)
    

    for d in ema_decays:
        k = str(d)
        ema_models[k].load_state_dict(checkpoint["ema_models"][k], strict=False)
    
    
    
    u.load_state_dict(checkpoint["net_model"], strict = False)
    
    optim.load_state_dict(checkpoint["optim"])
    
    sched.load_state_dict(checkpoint["sched"])
    init_step = checkpoint["step"]



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
            
        t_padded = pad_t_like_x(t, x0) 
        xt_data =   x1_data * t_padded + (1 - t_padded) * x0

        loss = F.mse_loss(x1_data - x0, u(t, xt_data, y))    
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(u.parameters(), grad_clip)  

        optim.step()
        sched.step()
            

        if logger is not None:
            logger.add_scalar(step, "FM loss", loss.detach().cpu().numpy())
                                     
        for d in ema_decays:
            ema(u, ema_models[str(d)], d)

        # sample and save the weights
        if save_image_step > 0 and step % save_image_step == 0:
                with torch.no_grad():
    
                    if COND:
                         y = torch.arange(num_save_image, device=device, dtype=int) % num_classes  
                    else:
                         y = None

                    generate_and_save_samples(u, savedir, y = y, name ="normal", one_step=False, batch_size=num_save_image, step = step, logger = logger)

                    for d in ema_decays:
                        tag = str(d).replace(".", "p")  # Safe for filenames
                        generate_and_save_samples(ema_models[str(d)], savedir, y = y, name =f"ema_{tag}", batch_size=num_save_image, step = step, logger = logger)


        if fid_step > 0 and (step+1) % fid_step == 0: 
                              
            if dataset_name == 'cifar10':
                fids.append(eval_cifar_fid(u, one_step=False, num_gen=num_gen, fid_batch_size=fid_batch_size))
            elif dataset_name == 'celeba':
                gen_dir = os.path.join(savedir, 'gen_celeba' )
                fids.append(eval_fid(u, gen_dir, args.celeba_dir, one_step=False, num_gen=num_gen, fid_batch_size=fid_batch_size))
    
            with open(os.path.join(savedir,  f"fids.pkl"), 'wb') as file:
                pickle.dump(fids, file)
                            
            if logger is not None:
                logger.add_scalar(step, "fid", fids[-1])

           
            for d in ema_decays:
                k = str(d)    
                if dataset_name == 'cifar10':
                    ema_fids[k].append(eval_cifar_fid(ema_models[k], one_step=False, num_gen=num_gen, fid_batch_size=fid_batch_size))
                elif dataset_name == 'celeba':
                    gen_dir = os.path.join(savedir, 'gen_celeba' )
                    ema_fids[k].append(eval_fid(ema_models[k], gen_dir, args.celeba_dir, one_step=False, num_gen=num_gen, fid_batch_size=fid_batch_size))
                            
                if logger is not None:
                    tag = k.replace(".", "p")
                    logger.add_scalar(step, f"ema_{tag}_fid", ema_fids[k][-1])

            # Save all EMA FID scores together
            with open(os.path.join(savedir, "ema_fids.pkl"), "wb") as file:
                pickle.dump(ema_fids, file)

            
            

        if save_model_step > 0 and step % save_model_step == 0:
            torch.save(
                {
                    "net_model": u.state_dict(),
                    "ema_models": {k: m.state_dict() for k, m in ema_models.items()},
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                },
                os.path.join(savedir, f"{dataset_name}_{args.cond}_FM_{step}.pt"),
            )
