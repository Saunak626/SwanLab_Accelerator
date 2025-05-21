import datetime
import src.datasets.base_dataset as datasets
from src.models.net import *

#=================== torch ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
#=================== accelerate ======================
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
#====================  swanlab =======================
import swanlab
from swanlab.integration.accelerate import SwanLabTracker


# 定义模型
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

# model.conv1 = nn.Conv2d(model.conv1.in_channels, model.conv1.out_channels, 3, 1, 1)
# model.maxpool = nn.Identity()
# model.fc = nn.Linear(model.fc.in_features, 10)

# model = models.resnet18(num_classes=10).to(device)

# print(f'initial model device: {next(model.parameters()).device}')



# 训练函数
def train(train_dataloader, model, loss_fn, accelerator, device, optimizer, lr_scheduler, epoch):
    # train model
    model.train()
    
    if accelerator.is_local_main_process:
        # print(f'{epoch}:{batch_idx}', accelerator.is_main_process, batch_idx, inputs.device, targets.device, next(model.parameters()).device)
        
        print(f"begin epoch {epoch} training...")
            
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        # print(f'{epoch}:{batch_idx}', accelerator.is_main_process, batch_idx, inputs.device, targets.device, model.device)
        
        print(f'{epoch}:{batch_idx}', accelerator.is_main_process, batch_idx, inputs.device, targets.device, next(model.parameters()).device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = criterion(outputs, targets)
        loss = loss_fn(outputs, targets)
        accelerator.backward(loss)
        
        optimizer.step()
        
        #################
        lr_scheduler.step() #TODO
        accelerator.log({"training_loss": loss, "epoch_num": epoch})
        ###################
        
        # 多卡训练时只在主进程打印信息，防止同一份信息打印好多遍
        if accelerator.is_local_main_process and batch_idx % 200 == 0:
            print(f'{accelerator.is_main_process}, Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_dataloader.dataset)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test(test_dataloader, model, loss_fn, accelerator, device):
    # eval model
    model.eval()
    # if accelerator.is_local_main_process:
    #     print(f"begin epoch {epoch} evaluating...")
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            
            # loss = criterion(outputs, targets)
            loss = loss_fn(outputs, targets)
            
            # gather data from multi-gpus (used when in ddp mode)
            test_loss += accelerator.gather(loss).sum().item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += accelerator.gather(pred.eq(targets.view_as(pred)).sum()).sum().item()
            #======================
            
            total += targets.size(0) * accelerator.num_processes  # 这里的总数可以直接累加
            
    test_loss /= total
    accuracy = 100. * correct / total
    
    if accelerator.is_local_main_process:
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)')

def main():
    set_seed(42)
    
    # hyperparameters
    config = {
        "num_epoch": 10,
        "batch_num": 2048,
        "learning_rate": 1e-4,
        "mixed_precision": "fp16", # fp16/bf16/no
    }

    BATCH_SIZE = config["batch_num"]
    lr=config["learning_rate"]
    epochs=config["num_epoch"]   
    
     
    # 初始化Accelerator
    accelerator = Accelerator()
    device = accelerator.device # accelerator自动维护的device

    # 只在主进程初始化tracker
    if accelerator.is_main_process:
        tracker = SwanLabTracker("CIFAR10_TRAING")
        accelerator = Accelerator(log_with=tracker)
        accelerator.init_trackers("CIFAR10_TRAING", config=config)

    # accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    # accelerator.print(f'device {str(accelerator.device)} is used!')





    # 加载数据集
    train_dataset = datasets.train_dataset_CIFAR10
    test_dataset = datasets.test_dataset_CIFAR10

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break    
    
    # 加载模型
    model = get_model()
    print(f'initial model device: {next(model.parameters()).device}')
    
    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=25*lr, epochs=epochs, steps_per_epoch=len(train_dataloader))

    # 使用accelerate包装模型、优化器和数据加载器
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, test_dataloader)

    # Get logger
    logger = get_logger(__name__)

    # 检查分配情况
    print(f"Model is on device: {next(model.parameters()).device}")
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        print(f"batch_{batch_idx} inputs are on device: {inputs.device}")
        print(f"batch_{batch_idx} targets are on device: {targets.device}")

    
    # 主训练循环
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n------")
        train(train_dataloader, model, loss_fn, accelerator, device, optimizer, lr_scheduler, epoch+1)
        test(test_dataloader, model, loss_fn, accelerator, device)


    accelerator.wait_for_everyone()  
    # 等所有卡都训练完了再保存checkpoints
    # accelerator.save_model(my_model, os.path.join("checkpoints", exp_name))

    accelerator.end_training()

    print("Training completed.")

if __name__ == "__main__":
    main()