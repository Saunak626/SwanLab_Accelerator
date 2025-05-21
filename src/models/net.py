
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import timm

from torch import nn
from torchinfo import summary
from torchvision import models


def get_model(): #获得预训练模型并冻住前面层的参数
    
    # model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = timm.create_model('resnet50', pretrained=True)
    
    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = True
    # 解冻分类头的参数
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    
    # 定义模型
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.conv1 = nn.Conv2d(model.conv1.in_channels, model.conv1.out_channels, 3, 1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    print(summary(model, input_size=(128, 3, 224, 224)))
    return model