import os
import datetime
import argparse # 新增: 导入 argparse 模块，用于解析命令行参数
# import src.datasets.base_dataset as datasets
from src.datasets.base_dataset import datasets_CIFAR10 as datasets_CIFAR10

import src.optimizers.optim as optimizers
from src.models.net import *
from tqdm import tqdm

#=================== torch ======================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#================= accelerate ===================
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
#================== swanlab =====================
import swanlab
from swanlab.integration.accelerate import SwanLabTracker

# 训练函数
def train(dataloader, model, loss_fn, accelerator, device, optimizer, lr_scheduler, epoch, batch_info=False):
    # train model
    model.train()
    
    # 仅在主进程创建进度条
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Training", unit="batch", position=0, leave=True)
            
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        # 根据参数打印batch处理信息
        if batch_info: 
            # 获取当前进程的本地排名
            local_rank = accelerator.state.local_process_index
            
            print(f"Process {local_rank}, Epoch: {epoch}, "
                  f"Batch Index: {batch_idx}, "
                  f"Inputs Device: {inputs.device}, "
                  f"Targets Device: {targets.device}, "
                  f"Model Device: {next(model.parameters()).device}")  
            
        # Compute prediction error
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
        lr_scheduler.step()
        
        accelerator.log({"train/loss": loss, "epoch_num": epoch})

        # 多卡训练时只在主进程打印信息
        if accelerator.is_main_process:
            progress_bar.update(1) # 更新进度条
            if batch_idx % 200 == 0:
                # 可以在进度条后附加信息
                progress_bar.set_postfix_str(f"Loss: {loss.item():.6f}")

    if accelerator.is_main_process:
        progress_bar.close() # 关闭进度条

# 测试函数
def test(dataloader, model, loss_fn, accelerator, device, epoch):
    model.eval()

    # 在本卡上累加的局部量
    local_test_loss_sum = torch.tensor(0.0, device=device)
    local_correct_sum = torch.tensor(0, device=device)
    local_sample_sum  = torch.tensor(0, device=device)

    # 仅在主进程创建测试进度条
    if accelerator.is_main_process:
        test_progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Testing ", unit="batch", position=0, leave=True)

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Compute prediction error
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            batch_size     = targets.size(0)
            sum_loss_batch = loss * batch_size
            correct_batch  = outputs.argmax(dim=1).eq(targets).sum()

            # 累加本卡
            local_test_loss_sum += sum_loss_batch
            local_correct_sum += correct_batch
            local_sample_sum  += batch_size
            
            # —— 按 batch 记录 loss & acc ——  
            batch_acc = 100. * correct_batch.item() / batch_size
            accelerator.log({
                "test/loss_batch": loss.item(),
                "test/acc_batch":  batch_acc,
            })  # 不传 step，就默认用全局 batch-step
    
            if accelerator.is_main_process:
                test_progress_bar.update(1) # 更新测试进度条
            
    # 全局聚合：收集所有GPU上的局部损失总和，一次性 gather & sum
    # total_test_loss = accelerator.gather(local_test_loss_sum).sum()
    # total_correct = accelerator.gather(local_correct_sum).sum()
    # total_samples = accelerator.gather(local_sample_sum).sum()

    # 全进程 all-reduce
    total_test_loss    = accelerator.reduce(local_test_loss_sum,  reduction="sum")
    total_correct = accelerator.reduce(local_correct_sum,    reduction="sum")
    total_samples = accelerator.reduce(local_sample_sum,     reduction="sum")

    # 只在主进程计算并打印
    if accelerator.is_main_process:
        avg_loss = (total_test_loss / total_samples).item()
        accuracy = 100. * total_correct.item() / total_samples.item()
        print(f'Epoch {epoch} '
              f'Test set: Average loss: {avg_loss:.4f}, '
              f'Accuracy: {total_correct}/{total_samples} ({accuracy:.0f}%)')
        
        accelerator.log({
        "test/loss": avg_loss,
        "test/accuracy": accuracy}, 
        step=epoch)
        
    
def main():
    parser = argparse.ArgumentParser(description="Python启动模式的指令配置")
    parser.add_argument(
        "--gpu_id",
        type=str, # gpu_id: 字符串类型，允许指定单个或多个GPU
        default='3', # default=None
        help="Specify the GPU ID(s) to use."
    )
    # 添加 --data_path 参数
    parser.add_argument(
        "--data_path",
        type=str,
        default='./data', # 默认值与 base_dataset.py 一致
        help="Specify the root directory for datasets."
    )
    # args: 解析后的命令行参数对象
    args, _ = parser.parse_known_args()

    set_seed(42)
    
    # hyperparameters
    config = {
        "num_epoch": 10, # 训练的总轮数
        "batch_num": 1024, # 每个批次的样本数量
        "learning_rate": 1e-4, # l初始学习率
        "mixed_precision": "fp16", # 训练精度 ('fp16'/'bf16'/'no')
        "optimizer": "adamW", # 使用的优化器名称
        "batch_info": False, # 是否打印详细的批处理信息的标志
    }
    
    # 从配置中获取参数
    BATCH_SIZE = config["batch_num"]
    lr = config["learning_rate"]
    epochs = config["num_epoch"]
    optimizer_name = config["optimizer"]
    mixed_precision = config["mixed_precision"]
    batch_info_flag = config["batch_info"]
    
    # 1. 判断是不是由 accelerate 启动（分布式）
    is_distributed = "LOCAL_RANK" in os.environ

    # 2. 如果是单卡，用 args.gpu_id 控制可见 GPU
    if not is_distributed:
        print(f"[INFO] Running with 'python train.py', setting CUDA_VISIBLE_DEVICES={args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 3. 解析全局 rank，判断主进程
    global_rank = int(os.environ.get("RANK", 0))
    is_main = (global_rank == 0)

    # 4. 根据分布式＆主进程决定要不要传 tracker 给 Accelerator
    
    # DeepSeed 模式下下必须要先初始化一次 Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    if is_main:
        # 主进程初始化tracker并创建带tracker的accelerator
        # 训练可视化 ,experiment_name=exp_name
        tracker = SwanLabTracker("CIFAR10_TRAING")
        accelerator = Accelerator(
            log_with=tracker,
        )
        # 初始化tracker
        accelerator.init_trackers("CIFAR10_TRAING", config=config)
        
    # 5. 打印确认
    print(f"Process {accelerator.local_process_index} using device: {accelerator.device}, "
          f"is_main: {accelerator.is_main_process}")
    
    device = accelerator.device # accelerator自动维护device
    
    # 加载数据集
    # train_dataset = datasets.train_dataset_CIFAR10
    # test_dataset = datasets.test_dataset_CIFAR10
    
    train_dataset, test_dataset = datasets_CIFAR10(root=args.data_path)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 获取并打印第一个样本
    X, y = next(iter(test_dataloader))
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    
    # 加载模型
    model = get_model()
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optimizers.get_optimizer(model, optimizer_name, lr) 
    
    # 定义学习率
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=25*lr, epochs=epochs, steps_per_epoch=len(train_dataloader)) 

    # 使用accelerate包装模型、优化器和数据加载器
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, test_dataloader)

    # Get logger 用处不明，似乎没有也能记录训练日志
    logger = get_logger(__name__)

    # 检查模型和数据的设备分配情况
    if batch_info_flag and accelerator.is_main_process: # 只在需要详细信息且是主进程时打印
        print(f"Model is on device: {next(model.parameters()).device}")
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            print(f"batch_{batch_idx} inputs are on device: {inputs.device}")
            print(f"batch_{batch_idx} targets are on device: {targets.device}")
            if batch_idx > 2: # 示例：只打印前几个批次
                break

    # 主训练循环
    for epoch_num in range(epochs):
        current_epoch = epoch_num + 1
        
        train(train_dataloader, model, loss_fn, accelerator, device, optimizer, lr_scheduler, current_epoch, batch_info=batch_info_flag) 
        test(test_dataloader, model, loss_fn, accelerator, device, current_epoch) 

    accelerator.wait_for_everyone()  
    # 等所有卡都训练完了再保存checkpoints
    # accelerator.save_model(my_model, os.path.join("checkpoints", exp_name))
    
    # unwrapped_model = accelerator.unwrap_model(model) 用哪个？
    accelerator.end_training()
    
    # =================================================
    # print logs and save ckpt  
    # accelerator.wait_for_everyone()
    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # accelerator.print(f"epoch【{epoch}】@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")
    # net_dict = accelerator.get_state_dict(model)
    # accelerator.save(net_dict,ckpt_path+"_"+str(epoch))
    
    # ==================================================
        
    print("Training completed.")

if __name__ == "__main__":
    main()