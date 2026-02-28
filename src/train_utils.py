import copy
import os

import torch
from torch import distributed as dist

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

try:
    import wandb
except Exception:
    print("Can not activate wandb")




from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )




def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x,y



class WandBWriter:
    # does not work
    def __init__(self, opt, savedir, exp_name):
        assert wandb.login(key=opt.wandb_api_key)
        wandb.init(dir=str(savedir), project="adv_fm", entity=opt.wandb_user, name=exp_name, config=vars(opt))

    def add_scalar(self, step, key, val):
        wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        # adopt from torchvision.utils.save_image
        image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        wandb.log({key: wandb.Image(image)}, step=step)


class TensorBoardWriter:
    def __init__(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
        self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        self.writer.close()


from torch.utils.data import Dataset
from torchvision.io import read_image   
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, ds_len, transform=None):
    
        self.img_dir = img_dir
        self.transform = transform
        self.len = ds_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx) + '.jpg')
        image = read_image(img_path)
       
        if self.transform:
            image = self.transform(image)
     
        return image, 0
    