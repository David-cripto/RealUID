
### Model with extra GAN Discriminator head
NUM_CLASSES = 1000

import torch.nn as nn
from torchcfm.models.unet.nn import timestep_embedding
from torchcfm.models.unet.unet import UNetModelWrapper

class UNetModelWrapperWithHead(UNetModelWrapper):
    def __init__(
        self,
        dim,
        num_channels,
        num_res_blocks,
        channel_mult=None,
        learn_sigma=False,
        class_cond=False,
        num_classes=NUM_CLASSES,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    ):
        
        super().__init__(
        dim,
        num_channels,
        num_res_blocks,
        channel_mult,
        learn_sigma,
        class_cond,
        num_classes,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        resblock_updown,
        use_fp16,
        use_new_attention_order,
        )


        self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=2, in_channels=256, out_channels=256, stride=2, padding=0),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.SiLU(),
                nn.Conv2d(kernel_size=2, in_channels=256, out_channels=256, stride=2, padding=0), 
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.SiLU(),
                nn.Conv2d(kernel_size=1, in_channels=256, out_channels=1, stride=1, padding=0), 
            ) 
        self.cls_pred_branch.requires_grad_(True)
            
        
    def forward(self, t, x, y=None, *args, **kwargs):
        return super().forward(t, x, y=y)

    def forward_head(self, t, x, y=None, *args, **kwargs):
        timesteps = t
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        while timesteps.dim() > 1:
            print(timesteps.shape)
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
  
        
        return self.cls_pred_branch(h)