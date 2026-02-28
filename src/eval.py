#Evaluate generator FID  
import torch
import numpy as np
from src.generate import gen_function, integrate_function
from cleanfid import fid
import shutil
import os
from torchvision.utils import  save_image


def get_model_device(model):
    return next(model.parameters()).device

def eval_cifar_fid(generator, one_step = True, num_gen = 50000, fid_batch_size = 100):
    def gen_1_img(unused_latent):
        with torch.no_grad():
            
            if generator.num_classes is not None:
                y = torch.arange(fid_batch_size, device=get_model_device(generator), dtype=int) % generator.num_classes 
            else:
                y = None
            
            z = torch.randn(fid_batch_size, 3, 32, 32, device=get_model_device(generator))

            if one_step:
                images = gen_function(generator, z, y)
            else:
                images = integrate_function(generator, z, y)

            images = (images.clip(-1, 1) * 127.5 + 128).clip(0, 255).to(torch.uint8)  
            return images

    generator.eval()
    print("Start computing FID")
    score = fid.compute_fid(
        gen=gen_1_img,
        dataset_name="cifar10",
        batch_size=fid_batch_size,
        dataset_res=32,
        num_gen=num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
        use_dataparallel=False, # to avoid error for NPU
    )
    print()
    print("FID has been computed")
    print()
    print("FID: ", score)
    generator.train()

    return score

# "./data_celeba/cut_celeba_full"

def eval_fid(generator, generation_folder, dataset_folder,  one_step = True,  num_gen = 50000, fid_batch_size = 100):
    
    
    total_img_num = 0
  
    if not os.path.exists(generation_folder):
        os.mkdir(generation_folder)
    else:
        shutil.rmtree(generation_folder)
        os.mkdir(generation_folder)
        
    def gen_1_img():
        nonlocal total_img_num 
        with torch.no_grad():
            z = torch.randn(fid_batch_size, 3, generator.image_size, generator.image_size, device=get_model_device(generator))
            
            if generator.num_classes is not None:
                y = torch.arange(fid_batch_size, device=get_model_device(generator), dtype=int) % generator.num_classes 
            else:
                y = None

            if one_step:
                images = gen_function(generator, z, y)
            else:
                images = integrate_function(generator, z, y)

            
            images = images.clip(-1, 1) / 2 + 0.5

            for i in range(fid_batch_size):
                img_path = os.path.join(generation_folder, f"{total_img_num}.jpg")
                total_img_num += 1
                save_image(images[i], img_path)

            return images
            
    for _ in range(num_gen//fid_batch_size):
        gen_1_img()
      
    

    generator.eval()

    print("Start computing FID")
    score = fid.compute_fid(dataset_folder,
                            generation_folder,
                            num_gen = num_gen,
                            use_dataparallel=False, # to avoid error for NPU
                            mode="legacy_tensorflow",
                            )
    print()
    print("FID has been computed")
    shutil.rmtree(generation_folder)
    print()
    print("FID: ", score)
    generator.train()
    return score