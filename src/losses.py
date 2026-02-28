import torch
from torchcfm.conditional_flow_matching import pad_t_like_x
import torch.nn.functional as F

#RealUID main distillation loss

def dist_loss(u, u_star, t, x0, x1_gen, x1_data, y = None, alpha = 1.0, beta = 1.0, generator_turn = True):
    
    t_padded = pad_t_like_x(t, x0) 
    batch_size = x0.shape[0]
    xt_gen =   x1_gen * t_padded + (1 - t_padded) * x0
    u_xt_gen = u(t, xt_gen, y)
        
    if not generator_turn:
        xt_data =  t_padded * x1_data + (1 - t_padded) * x0
        u_xt_data = u(t, xt_data, y)
        
        loss = - alpha * torch.sum((u_xt_gen - beta/alpha * (x1_gen - x0))**2)
        
        if alpha != 1.0:
            loss = loss - (1.0 - alpha) * torch.sum((u_xt_data - (1.0 - beta)/( 1.0 - alpha) * (x1_data - x0))**2)
        else:
            loss = loss + 2 * (1.0 - beta) * torch.sum(u_xt_data * (x1_data - x0))
    else:
        u_star_xt_gen = u_star(t, xt_gen, y)
        loss = alpha * torch.sum((u_star_xt_gen - beta/alpha * (x1_gen - x0))**2) - alpha * torch.sum((u_xt_gen - beta/alpha * (x1_gen - x0))**2)
    
    return  loss/batch_size



def general_dist_loss(u, u_star, t, x0, x1_gen, x1_data, y = None, alpha = 1.0, beta = 1.0, gamma = 1.0, parametrization = 'standard', generator_turn = True):
    
    t_padded = pad_t_like_x(t, x0) 
    batch_size = x0.shape[0]

    
    xt_gen =   x1_gen * t_padded + (1 - t_padded) * x0
    u_xt_gen = u(t, xt_gen, y)
    u_star_xt_gen = u_star(t, xt_gen, y)

    if parametrization == 'standard':
        delta_xt_gen = u_xt_gen - u_star_xt_gen
    elif parametrization == 'beta':
        delta_xt_gen = beta * (u_xt_gen - u_star_xt_gen)


    loss = - alpha * torch.sum(delta_xt_gen * delta_xt_gen) \
        + 2 * beta * torch.sum(delta_xt_gen * (x1_gen - x0))\
        - 2 * alpha * torch.sum(delta_xt_gen *  u_star_xt_gen)

    if not generator_turn:
        xt_data =  t_padded * x1_data + (1 - t_padded) * x0
        u_xt_data = u(t, xt_data, y)
        u_star_xt_data = u_star(t, xt_data, y)
        if parametrization == 'standard':
            delta_xt_data = u_xt_data - u_star_xt_data
        elif parametrization == 'beta':
            delta_xt_data = beta * (u_xt_data - u_star_xt_data)

        loss = loss - (1 - alpha) * torch.sum(delta_xt_data * delta_xt_data) \
            + 2 * (1 - beta) * torch.sum(delta_xt_data * (x1_data - x0))\
            - 2 * (1 - alpha) * torch.sum(delta_xt_data *  u_star_xt_data)

    return  loss/batch_size


# GAN loss
    
def compute_cls_logits(u, x, t, y = None):
    logits = u.forward_head(t, x, y).float()
    return logits
    

def GANloss(u,  t, x0, x1_gen, x1_data, y = None, generator_turn = True):
    
    
    t_padded = pad_t_like_x(t, x0)
    xt_gen =   x1_gen * t_padded + (1 - t_padded) * x0
    xt_data =   x1_data * t_padded + (1 - t_padded) * x0
    

    if generator_turn:
        pred_realism_on_fake_with_grad = compute_cls_logits(u, xt_gen, t, y)
        return F.softplus(-pred_realism_on_fake_with_grad).mean()
    else:
        pred_realism_on_real = compute_cls_logits(
            u, xt_data.detach(), t, y
        )
        pred_realism_on_fake = compute_cls_logits(
            u, xt_gen.detach(), t, y
        )
        return  -F.softplus(pred_realism_on_fake).mean() - F.softplus(-pred_realism_on_real).mean()