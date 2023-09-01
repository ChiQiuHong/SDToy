from ldm.modules.attention import CrossAttention
from ldm.modules.diffusionmodules.openaimodel import UNetModel

import torch

def TestCrossattention():
    attn_cls = CrossAttention(96, 96).cuda()

    # b c h w -> b (h w) c
    x = torch.rand(1, 16384, 96).cuda()

    output = attn_cls(x)

    print(output.shape)


def TestUNetModel():
    unet = UNetModel().cuda()

    latent = torch.randn(1, 4, 64, 64).cuda()
    text_embeddings = torch.randn(1, 77, 768).cuda()
    timestamp = torch.tensor([0]).cuda()

    output = unet(latent, timestamp, text_embeddings)

    print(output.shape)