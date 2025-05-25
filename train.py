import os
import datetime
import argparse
import re
import sys
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

# 简化的检查点工具函数
def find_latest_checkpoint_in_dir(directory):
    """在指定目录中查找最新的检查点"""
    if not os.path.isdir(directory):
        return None, -1
    
    latest_epoch = -1
    latest_checkpoint = None
    
    for name in os.listdir(directory):
        if name.startswith("epoch_") and name.endswith("_checkpoint"):
            match = re.search(r"epoch_(\d+)_checkpoint", name)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = os.path.join(directory, name)
    
    return latest_checkpoint, latest_epoch

def find_global_latest_checkpoint(base_dir):
    """在整个结果目录中查找全局最新检查点"""
    if not os.path.isdir(base_dir):
        return None, -1, None
    
    global_latest_epoch = -1
    global_latest_checkpoint = None
    global_experiment_dir = None
    
    for exp_name in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_name)
        if os.path.isdir(exp_path) and exp_name.startswith("CIFAR10_"):
            checkpoint, epoch = find_latest_checkpoint_in_dir(exp_path)
            if checkpoint and epoch > global_latest_epoch:
                global_latest_epoch = epoch
                global_latest_checkpoint = checkpoint
                global_experiment_dir = exp_path
    
    return global_latest_checkpoint, global_latest_epoch, global_experiment_dir

def load_checkpoint_and_setup_experiment(accelerator, args, default_exp_dir, default_exp_name):
    """加载检查点并设置实验参数"""
    if not args.resume:
        # 不恢复，使用默认设置
        if accelerator.is_main_process:
            os.makedirs(default_exp_dir, exist_ok=True)
        return 0, default_exp_dir, default_exp_name
    
    resume_value = args.resume
    checkpoint_path = None
    start_epoch = 0
    final_exp_dir = default_exp_dir
    final_exp_name = default_exp_name
    
    # 处理不同的恢复选项
    if resume_value == "latest":
        checkpoint_path, start_epoch, final_exp_dir = find_global_latest_checkpoint(args.output)
        if checkpoint_path:
            final_exp_name = os.path.basename(final_exp_dir)
            accelerator.print(f"找到全局最新检查点: {checkpoint_path}")
        else:
            accelerator.print(f"在 {args.output} 中未找到检查点，程序退出")
            sys.exit(1)
    
    elif os.path.isdir(resume_value):
        if resume_value.endswith("_checkpoint"):
            # 具体检查点目录
            checkpoint_path = resume_value
            match = re.search(r"epoch_(\d+)_checkpoint", os.path.basename(resume_value))
            start_epoch = int(match.group(1)) if match else 0
            final_exp_dir = os.path.dirname(resume_value)
            final_exp_name = os.path.basename(final_exp_dir)
        elif os.path.basename(resume_value).startswith("CIFAR10_"):
            # 实验目录
            checkpoint_path, start_epoch = find_latest_checkpoint_in_dir(resume_value)
            if checkpoint_path:
                final_exp_dir = resume_value
                final_exp_name = os.path.basename(resume_value)
                accelerator.print(f"在实验 {final_exp_name} 中找到检查点: {checkpoint_path}")
            else:
                accelerator.print(f"在实验目录 {resume_value} 中未找到检查点，程序退出")
                sys.exit(1)
    
    if not checkpoint_path:
        accelerator.print(f"无效的 --resume 值: {resume_value}，程序退出")
        sys.exit(1)
    
    # 加载检查点
    try:
        accelerator.print(f"正在加载检查点: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        accelerator.print(f"成功加载检查点，将从第 {start_epoch + 1} 轮开始训练")
    except Exception as e:
        accelerator.print(f"加载检查点失败: {e}，程序退出")
        sys.exit(1)
    
    # 确保实验目录存在
    if accelerator.is_main_process:
        os.makedirs(final_exp_dir, exist_ok=True)
    
    return start_epoch, final_exp_dir, final_exp_name

def save_checkpoint_if_needed(accelerator, args, current_epoch, total_epochs, experiment_dir):
    """根据设置保存检查点"""
    should_save = (current_epoch % args.save_freq == 0) or (current_epoch == total_epochs)
    
    if should_save:
        checkpoint_dir = os.path.join(experiment_dir, f"epoch_{current_epoch}_checkpoint")
        accelerator.save_state(checkpoint_dir)
        
        if accelerator.is_main_process:
            accelerator.print(f"已保存第 {current_epoch} 轮检查点: {checkpoint_dir}")

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
    # 添加模型保存和断点续训相关参数
    parser.add_argument(
        "--output",
        type=str,
        default="./result",
        help="Base directory to save experiment results."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None, # None表示不恢复，"latest"表示从最新的恢复
        help="Path to checkpoint directory to resume from, or 'latest' to resume from the most recent checkpoint."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1, # 默认每1个epoch保存一次
        help="Save a checkpoint every N epochs."
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
    total_epochs = config["num_epoch"] # 修改变量名以区分
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
    is_main_process_global = (global_rank == 0) # accelerator.is_main_process 前的判断

    # 4. 根据分布式＆主进程决定要不要传 tracker 给 Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    # 初始的实验名称和目录（如果不是从检查点恢复，则使用这个）
    initial_experiment_name = "CIFAR10_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    initial_experiment_dir = os.path.join(args.output, initial_experiment_name)

    # 加载检查点并确定最终的实验设置
    start_epoch_idx, experiment_dir, experiment_name = load_checkpoint_and_setup_experiment(
        accelerator, args, initial_experiment_dir, initial_experiment_name
    )

    # 如果是恢复训练，并且SwanLab tracker需要重新初始化，则根据恢复的实验名进行
    # 注意：Accelerator的log_with通常在初始化时设置，如果恢复到不同实验，tracker需要重新配置
    # 简单的做法是，如果experiment_name改变了（意味着从其他实验恢复），则重新初始化带tracker的accelerator
    if accelerator.is_main_process:
        # 只有当experiment_name是从旧的恢复而来，且与新生成的不一致时，才可能需要重新设置tracker
        # 但由于我们总是基于experiment_name创建tracker, 所以这里保持不变，确保tracker与experiment_name一致
        tracker = SwanLabTracker(experiment_name) # 使用最终的实验名称
        accelerator = Accelerator(
            log_with=tracker,
            mixed_precision=mixed_precision,
        )
        # 初始化tracker
        accelerator.init_trackers(experiment_name, config=config)
        
    # 5. 打印确认
    accelerator.print(f"Process {accelerator.local_process_index} using device: {accelerator.device}, "
                      f"is_main: {accelerator.is_main_process}")
    accelerator.print(f"Experiment: {experiment_name}")
    accelerator.print(f"Results will be saved to: {experiment_dir}")
    
    device = accelerator.device # accelerator自动维护device
    
    # 加载数据集
    train_dataset, test_dataset = datasets_CIFAR10(root=args.data_path)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 获取并打印第一个样本
    if accelerator.is_main_process: # 只在主进程打印样本信息
        X, y = next(iter(test_dataloader))
        accelerator.print(f"Shape of X [N, C, H, W]: {X.shape}")
        accelerator.print(f"Shape of y: {y.shape} {y.dtype}")
    
    # 加载模型
    model = get_model()
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optimizers.get_optimizer(model, optimizer_name, lr) 
    
    # 定义学习率
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=25*lr, epochs=total_epochs, steps_per_epoch=len(train_dataloader)) 

    # 使用accelerate包装模型、优化器和数据加载器
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    # Get logger
    logger = get_logger(__name__)

    # 检查模型和数据的设备分配情况
    if batch_info_flag and accelerator.is_main_process: # 只在需要详细信息且是主进程时打印
        accelerator.print(f"Model is on device: {next(model.parameters()).device}")
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            print(f"batch_{batch_idx} inputs are on device: {inputs.device}")
            print(f"batch_{batch_idx} targets are on device: {targets.device}")
            if batch_idx > 2: # 示例：只打印前几个批次
                break

    # 主训练循环
    accelerator.print(f"Starting training from epoch {start_epoch_idx + 1} up to {total_epochs} epochs.")
    for epoch_idx in range(start_epoch_idx, total_epochs):
        current_epoch = epoch_idx + 1 # 当前是第几轮 (1-indexed)
        
        accelerator.print(f"--- Starting Epoch {current_epoch} ---")
        train(train_dataloader, model, loss_fn, accelerator, device, optimizer, lr_scheduler, current_epoch, batch_info=batch_info_flag) 
        test(test_dataloader, model, loss_fn, accelerator, device, current_epoch) 

        # 保存检查点
        save_checkpoint_if_needed(accelerator, args, current_epoch, total_epochs, experiment_dir)

    accelerator.wait_for_everyone()  
    
    # 保存最终模型 (仅主进程)
    if accelerator.is_main_process:
        final_model_path = os.path.join(experiment_dir, "final_model")
        accelerator.save_model(model, final_model_path) # model是prepare后的模型，accelerator会处理unwrap
        accelerator.print(f"Saved final model at {final_model_path}")
    
    accelerator.end_training()
    
    accelerator.print("Training completed.")

if __name__ == "__main__":
    main()