import datetime

#=================== torch ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
#=================== accelerate ======================
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
#====================  swanlab =======================
import swanlab
from swanlab.integration.accelerate import SwanLabTracker



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
# accelerator = Accelerator()
# device = accelerator.device # accelerator自动维护的device

# 先初始化accelerator，再附加tracker
accelerator = Accelerator()
device = accelerator.device

# 只在主进程初始化tracker
if accelerator.is_main_process:
    tracker = SwanLabTracker("CIFAR10_TRAING")
    accelerator = Accelerator(log_with=tracker)
    accelerator.init_trackers("CIFAR10_TRAING", config=config)

# accelerator = Accelerator(mixed_precision=config["mixed_precision"])
# accelerator.print(f'device {str(accelerator.device)} is used!')

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 定义模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(model.conv1.in_channels, model.conv1.out_channels, 3, 1, 1)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

# model = models.resnet18(num_classes=10).to(device)

print(f'initial model device: {next(model.parameters()).device}')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=25*lr, epochs=epochs, steps_per_epoch=len(train_loader))

# 使用accelerate包装模型、优化器和数据加载器
model, optimizer, lr_scheduler, train_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, test_loader)

# Get logger
logger = get_logger(__name__)

# 检查分配情况
print(f"Model is on device: {next(model.parameters()).device}")
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"batch_{batch_idx} inputs are on device: {inputs.device}")
    print(f"batch_{batch_idx} targets are on device: {targets.device}")



# 训练函数
def train(epoch):
    # train model
    model.train()
    if accelerator.is_local_main_process:
        # print(f'{epoch}:{batch_idx}', accelerator.is_main_process, batch_idx, inputs.device, targets.device, next(model.parameters()).device)
        
        print(f"begin epoch {epoch} training...")
            
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(f'{epoch}:{batch_idx}', accelerator.is_main_process, batch_idx, inputs.device, targets.device, model.device)
        
        print(f'{epoch}:{batch_idx}', accelerator.is_main_process, batch_idx, inputs.device, targets.device, next(model.parameters()).device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        
        optimizer.step()
        
        #################
        lr_scheduler.step() #TODO
        accelerator.log({"training_loss": loss, "epoch_num": epoch})
        ###################
        
        # 多卡训练时只在主进程打印信息，防止同一份信息打印好多遍
        if accelerator.is_local_main_process and batch_idx % 200 == 0:
            print(f'{accelerator.is_main_process}, Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test(epoch):
    # eval model
    model.eval()
    if accelerator.is_local_main_process:
        print(f"begin epoch {epoch} evaluating...")
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
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

# 主训练循环
for epoch in range(epochs):
    train(epoch)
    test(epoch)

print("Training completed.")