#!/bin/bash

dataset=$1

# Run the code below in command window to set CUDA visible devices and run specific script
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#sh run.sh 'cifar10-uncond-teacher' 



#Set up the output folder in --output_dir and --exp_name if needed

# Decrease --batch_size and --fid_batch_size to reduce memory consumption

# Add extra ema decays to track --ema_decays 

# Add the logger option:
# --logger_name wandb  --wandb_user 'user'  --wandb_api_key 'key' 
# or --logger_name tensorboard

# Add the option to any run below to load a checkpoint:
# --resume_ckpt_path 'exp_name.pt'

if [ "$dataset" = 'cifar10-uncond-teacher' ]; then

    python3 train_teacher.py \
    --lr 2e-4\
    --batch_size 256 \
    --ema_decays 0.999 0.9996 0.9999 \
    --output_dir ./results \
    --dataset cifar10 \
    --cond uncond \
    --save_image_step 5000 \
    --num_save_image 100 \
    --fid_step 20000 \
    --save_model_step 20000


elif [ "$dataset" = 'cifar10-cond-teacher' ]; then

    python3 train_teacher.py \
    --lr 2e-4\
    --batch_size 256 \
    --ema_decays 0.999 0.9996 0.9999 \
    --output_dir ./results \
    --dataset cifar10 \
    --cond cond \
    --save_image_step 5000 \
    --num_save_image 100 \
    --fid_step 20000 \
    --save_model_step 20000 

elif [ "$dataset" = 'celeba-teacher' ]; then

    python3 train_teacher.py \
    --lr 4e-5\
    --batch_size 256 \
    --ema_decays 0.999 0.9996 0.9999 \
    --output_dir ./results \
    --dataset celeba \
    --celeba_dir data/cut_celeba_full \
    --save_image_step 5000 \
    --num_save_image 36 \
    --fid_step 20000 \
    --save_model_step 20000 


elif [ "$dataset" = 'cifar10-uncond-distil' ]; then

    #Vary coefs --alpha, --beta from 0.85 to 1.00. Better keep the ratio beta/alpha = 1.02 or 0.98. 
    #The case alpha = beta = 1.0 is data-free.
    # Add the extra coef gamma:
    # --gamma 0.92

    python3 train_distil.py \
    --alpha 0.92 \
    --beta 0.94 \
    --batch_size 256 \
    --lr 3e-5\
    --ema_decays 0.999 0.9996 0.9999 \
    --teacher_ckpt_path ./model_checkpoints/cifar10_uncond_FM_400000.pt \
    --output_dir ./results \
    --dataset cifar10 \
    --cond uncond \
    --save_image_step 5000 \
    --num_save_image 100 \
    --fid_step 5000 \
    --save_model_step 10000 

elif [ "$dataset" = 'cifar10-cond-distil' ]; then

    #Vary coefs --alpha, --beta from 0.85 to 1.00. Better keep the ratio beta/alpha = 1.02 or 0.98. 
    #The case alpha = beta = 1.0 is data-free.
    # Add the extra coef gamma:
    # --gamma 0.98

    python3 train_distil.py \
    --alpha 0.98 \
    --beta 0.96 \
    --batch_size 256 \
    --lr 3e-5\
    --ema_decays 0.999 0.9996 0.9999 \
    --teacher_ckpt_path ./model_checkpoints/cifar10_cond_FM_400000.pt \
    --output_dir ./results \
    --dataset cifar10 \
    --cond cond \
    --save_image_step 5000 \
    --num_save_image 100 \
    --fid_step 5000 \
    --save_model_step 10000 

elif [ "$dataset" = 'celeba-distil' ]; then

    #Vary coefs --alpha, --beta from 0.85 to 1.00. Better keep the ratio beta/alpha = 1.02 or 0.98. 
    #The case alpha = beta = 1.0 is data-free.
    # Add an extra coef gamma:
    # --gamma 0.98

    python3 train_distil.py \
    --alpha 0.90 \
    --beta 0.88 \
    --batch_size 64 \
    --lr  5e-6 \
    --ema_decays 0.999 0.9996 0.9999 \
    --total_steps 1600000\
    --teacher_ckpt_path ./model_checkpoints/celeba_uncond_FM_239999.pt \
    --output_dir ./results \
    --dataset celeba \
    --celeba_dir data/cut_celeba_full \
    --save_image_step 5000 \
    --num_save_image 36 \
    --fid_step 5000 \
    --save_model_step 10000 

elif [ "$dataset" = 'cifar10-uncond-distil-gan' ]; then

    #Vary gan coefs --gen_coef and --disc_coef. We keep the same ration disc_coef/gen_coef = 3.

    #gen_coef =  5.0 # 0.3 1.0 5.0 25.0  the best set is (5.0, 15.0)
    #disc_coef = 15.0 # 1.0 3.0 15.0 75.0 

    #Optionally, one can combibe RealUID and GANs by setting --alpha and --beta not equal to 1.0.

    python3 train_distil.py \
    --alpha 1.00 \
    --beta 1.00 \
    --with_gan_loss \
    --gen_coef 0.3\
    --disc_coef 1.0\
    --batch_size 256 \
    --lr 3e-5\
    --ema_decays 0.999 0.9996 0.9999 \
    --teacher_ckpt_path ./model_checkpoints/cifar10_uncond_FM_400000.pt \
    --output_dir ./results \
    --dataset cifar10 \
    --cond uncond \
    --save_image_step 5000 \
    --num_save_image 100 \
    --fid_step 5000 \
    --save_model_step 10000 

elif [ "$dataset" = 'cifar10-cond-distil-gan' ]; then

    #Vary gan coefs --gen_coef and --disc_coef. We keep the same ration disc_coef/gen_coef = 3.

    #gen_coef =  5.0 # 0.3 1.0 5.0 25.0  the best set is (5.0, 15.0)
    #disc_coef = 15.0 # 1.0 3.0 15.0 75.0 

    #Optionally, one can combibe RealUID and GANs by setting --alpha and --beta not equal to 1.0.

    python3 train_distil.py \
    --alpha 1.00 \
    --beta 1.00 \
    --with_gan_loss \
    --gen_coef 0.3\
    --disc_coef 1.0\
    --batch_size 256 \
    --lr 3e-5\
    --ema_decays 0.999 0.9996 0.9999 \
    --teacher_ckpt_path ./model_checkpoints/cifar10_cond_FM_400000.pt \
    --output_dir ./results \
    --dataset cifar10 \
    --cond cond \
    --save_image_step 5000 \
    --num_save_image 100 \
    --fid_step 5000 \
    --save_model_step 10000 

elif [ "$dataset" = 'celeba-distil-gan' ]; then

    #Vary gan coefs --gen_coef and --disc_coef. We keep the same ration disc_coef/gen_coef = 3.

    #gen_coef =  5.0 # 0.3 1.0 5.0 25.0  the best set is (5.0, 15.0)
    #disc_coef = 15.0 # 1.0 3.0 15.0 75.0 

    #Optionally, one can combibe RealUID and GANs by setting --alpha and --beta not equal to 1.0.

    python3 train_distil.py \
    --alpha 1.00 \
    --beta 1.00 \
    --with_gan_loss \
    --gen_coef 1.0\
    --disc_coef 3.0\
    --batch_size 64 \
    --lr  5e-6 \
    --ema_decays 0.999 0.9996 0.9999 \
    --total_steps 1600000\
    --teacher_ckpt_path ./model_checkpoints/celeba_uncond_FM_239999.pt \
    --output_dir ./results \
    --dataset celeba \
    --celeba_dir data/cut_celeba_full \
    --save_image_step 5000 \
    --num_save_image 36 \
    --fid_step 5000 \
    --save_model_step 10000 

elif [ "$dataset" = 'cifar10-uncond-distil-finetune' ]; then

    #Name the best distilled checkpoint 'cifar10_uncond_distil.pt' and save it to model_checkpoints folder 
    #Vary coefs --alpha, --beta from 0.85 to 1.00. 
    #Better keep the ratio beta/alpha = 1.06 or 0.94. 

    python3 train_distil.py \
    --alpha 0.92 \
    --beta 0.86 \
    --batch_size 256 \
    --lr 1e-5\
    --warmup 1\
    --ema_decays 0.999 0.9996 0.9999 \
    --finetune_ckpt_path ./model_checkpoints/cifar10_uncond_distil.pt\
    --finetune_ema 0.999\
    --teacher_ckpt_path ./model_checkpoints/cifar10_uncond_FM_400000.pt \
    --output_dir ./results \
    --dataset cifar10 \
    --cond uncond \
    --save_image_step 2000 \
    --num_save_image 100 \
    --fid_step 2000 \
    --save_model_step 2000 

elif [ "$dataset" = 'cifar10-cond-distil-finetune' ]; then

    #Name the best distilled checkpoint 'cifar10_cond_distil.pt' and save it to model_checkpoints folder 
    #Vary coefs --alpha, --beta from 0.85 to 1.00. 
    #Better keep the ratio beta/alpha = 1.06 or 0.94. 

    python3 train_distil.py \
    --alpha 0.94 \
    --beta 1.00 \
    --batch_size 256 \
    --lr 1e-5\
    --warmup 1\
    --ema_decays 0.999 0.9996 0.9999 \
    --finetune_ckpt_path ./model_checkpoints/cifar10_cond_distil.pt\
    --finetune_ema 0.9999\
    --teacher_ckpt_path ./model_checkpoints/cifar10_cond_FM_400000.pt \
    --output_dir ./results \
    --dataset cifar10 \
    --cond cond \
    --save_image_step 2000 \
    --num_save_image 100 \
    --fid_step 2000 \
    --save_model_step 2000 
fi
