import torch 
import numpy as np
from torchvision.utils import make_grid, save_image
from torchdyn.core import NeuralODE
import os
import torch.nn as nn

def get_model_device(model):
    return next(model.parameters()).device

class Wrapper(nn.Module):
    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        self.y = y

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, y=self.y)
    
def integrate_function(u, z, y = None):
    vf = Wrapper(u, y)   
    

    node_ = NeuralODE(vf, solver="euler", sensitivity="adjoint")
    t_span = torch.linspace(0, 1, 100, device=z.device, dtype=torch.float64)
    with torch.no_grad():
        traj = node_.trajectory(
            z,
            t_span=t_span,
        )
        images = traj[-1, :].view([-1, 3, u.image_size, u.image_size])
    
    return images

def gen_function(generator, z, y = None):
    return z + generator(torch.zeros(z.shape[0]).type_as(z), z, y)


def generate_and_save_samples(generator, savedir, y = None, one_step = True, batch_size = 36, name ="",  step = 0, logger = None):
    generator.eval()
    with torch.no_grad():
        
         
        z = torch.randn(batch_size, 3,  generator.image_size, generator.image_size, device=get_model_device(generator))
        
        if one_step:
            images = gen_function(generator, z, y)
        else:
            images = integrate_function(generator, z, y)


        images = images.clip(-1,1) / 2 + 0.5

    img_path = os.path.join(savedir, f"{name}_{step}.png")

    nrow = int(np.sqrt(batch_size))

    save_image(images, img_path, nrow=nrow)

    if logger is not None:
        logger.add_image(step, f"{name}_{step}", make_grid(images, nrow=nrow))
        
    generator.train()