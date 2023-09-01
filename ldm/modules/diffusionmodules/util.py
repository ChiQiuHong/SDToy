import math
import torch
import torch.nn as nn
from einops import repeat


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    通常在计算梯度的时候，我们需要将所有前向传播的激活值保存下来，zhe
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
    

class CheckpointFunction(torch.autograd.Function):
    """
    1 forward pass时, 直接运行模型的forward函数, 不缓存中间变量, 返回输出
    2 backward pass时, 重新运行forward函数, 这次把中间变量缓存下来, 然后调用
      autograd.grad计算梯度
    这样避免了正常训练时要缓存大量中间变量的问题, 降低GPU内存占用
    """
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    
    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)

        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.module for normalization.
    """
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    """
    nn.GroupNorm 是一种归一化技术，它将输入张量分成组，并在每个组内进行归一化。
    这对于小批量大小的情况以及在通道数较少的情况下特别有用，
    因为它可以在这些情况下提供与批量归一化类似的正则化效果，同时不会受到小批量大小的限制。
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    return ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings. 时间步数采用正余弦编码
    时间步长嵌入在序列模型中常用于将时间信息引入模型，以帮助模型理解序列数据中的时间关系。
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minium frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果嵌入维度是奇数，添加一列零以匹配维度
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


if __name__ == "__main__":
    timesteps = torch.arange(0, 10) 

    embedding = timestep_embedding(timesteps, dim=5)

    print("\nTime steps:")
    print(timesteps)
    print("\nTime step embedding:")
    print(embedding)