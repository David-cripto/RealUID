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




import os
import pickle
import numpy as np



from src.models import UNetModelWrapperWithHead
from src.eval import eval_cifar_fid, eval_fid

  

parser = argparse.ArgumentParser(description="model parser")



#paths
parser.add_argument('--output_dir', type=str) #"./result_cifar/"
parser.add_argument('--exp_name', type=str, default=None, help='name of the subfolder in the output dir') 

parser.add_argument('--model_ckpt_path', type=str, help = 'model checkpoint')
parser.add_argument('--mode',default='one_step',  choices=['one_step', 'multi_step'], help='one-step mode or flow vector field')
parser.add_argument('--ema_decay', type=float, default=None, help='ema decay which to evaluate if given')
#dataset
parser.add_argument('--dataset', default='cifar10',  choices=['cifar10', 'celeba'])
parser.add_argument('--cond', default='uncond',  choices=['cond', 'uncond'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--celeba_dir', type=str, default=None, help='folder with celeba dataset images if chosen') #./data_celeba/cut_celeba_full"
parser.add_argument('--num_classes', type=int, default=10)


parser.add_argument('--num_gen', type=int, default=50_000, help = 'number of generated samples for fid evaluation, do not change') 
parser.add_argument('--fid_batch_size', type=int, default=100, help ='batch size for evaluating fid, it must be divisible by the number of classes and must divide num_gen if conditional') 


args = parser.parse_args()




COND = args.cond == 'cond' #conditional or unconditional
num_classes = args.num_classes

fid_batch_size = args.fid_batch_size
num_gen = args.num_gen
dataset_name = args.dataset
one_step = args.mode == 'one_step'
ema_decay = args.ema_decay

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



  
exp_name = args.exp_name
if exp_name is None:
    exp_name = f'{args.dataset}_{args.cond}_eval_fid'

output_dir = args.output_dir


savedir = os.path.join(output_dir, exp_name)
os.makedirs(savedir, exist_ok=True)


###Init allmodels

resolution = 32 if dataset_name == 'cifar10' else 64

model = UNetModelWrapperWithHead(
        dim=(3, resolution, resolution),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.0,
        class_cond = COND,
        num_classes = num_classes
    ).to(device)  



print("path: ", args.model_ckpt_path)
checkpoint = torch.load(args.model_ckpt_path, map_location=device)

if one_step:
    if ema_decay is None:
        model.load_state_dict(checkpoint["gen"], strict = False)
    else:
        model.load_state_dict(checkpoint["ema_gens"][str(ema_decay)], strict = False)
else:
    if ema_decay is None:
        model.load_state_dict(checkpoint["net_model"], strict = False)
    else:
        model.load_state_dict(checkpoint["ema_models"][str(ema_decay)], strict = False)
     

model.eval()
  
fids = []


with torch.no_grad():          
        
    if dataset_name == 'cifar10':
        fids.append(eval_cifar_fid(model, one_step=one_step, num_gen=num_gen, fid_batch_size = fid_batch_size))
    elif dataset_name == 'celeba':
        gen_dir = os.path.join(savedir, 'gen_celeba')
        fids.append(eval_fid(model, gen_dir, args.celeba_dir, one_step=one_step, num_gen=num_gen, fid_batch_size = fid_batch_size))
                       
                            
    with open(os.path.join(savedir,  f"fids.pkl"), 'wb') as file:
        pickle.dump(fids, file)
                            
                        

           
