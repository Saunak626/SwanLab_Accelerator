import torch

def get_optimizer(net, optim, lr):
    #优化器选择
    if optim == "sgd": 
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=0)
    elif optim == "adam":
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=0)
    elif optim == "adamW":
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=0)

    return optimizer