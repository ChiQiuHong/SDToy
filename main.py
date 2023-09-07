from ldm.modules.diffusionmodules.openaimodel import UNetModel

from ldm.modules.test import *

import torch

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    print(out)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

if __name__ == "__main__":
    # TestUNetModel()

    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.float32)
    t = torch.tensor([6], dtype=torch.long)
    x_shape = (3, 5)
    output = extract_into_tensor(a, t, x_shape)
    print(output)