import torch.nn as nn
from models.module.Attn import *

POOL_LAYER = {
    "no": nn.Identity(),
    "avg": nn.AvgPool2d(2, 2),
    "max": nn.MaxPool2d(2, 2)
}

ATTN_LAYER = {
    "no": nn.Identity,
    "ASA": ASA,
}
