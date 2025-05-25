
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import timm

from torch import nn
from torchinfo import summary
from torchvision import models


def get_model(): #获得预训练模型并冻住前面层的参数
    # 定义模型
    
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = timm.create_model('resnet50', pretrained=True)
    
    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = True
    # 解冻分类头的参数
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    
    model.conv1 = nn.Conv2d(model.conv1.in_channels, model.conv1.out_channels, 3, 1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # print(summary(model, input_size=(128, 3, 224, 224))
    
    # print(summary(model, input_size=(128, 3, 32, 32), verbose=2))
    return model

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将模型移至指定设备
    model = get_model().to(device)
    
    # 创建输入张量为 [批量大小, 通道数, 高度, 宽度]
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印输入和输出的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    