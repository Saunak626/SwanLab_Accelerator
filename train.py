import os
import datetime
import argparse # 新增: 导入 argparse 模块，用于解析命令行参数
import src.datasets.base_dataset as datasets
import src.optimizers.optim as optimizers
from src.models.net import *
from tqdm import tqdm # 导入tqdm

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
    if accelerator.is_local_main_process:
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Training", unit="batch")
            
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_info: # 根据参数决定是否打印批处理信息
            # 获取当前进程的本地排名
            local_rank = accelerator.state.local_process_index
            
            print(f"Process {local_rank}, Epoch: {epoch}, Batch Index: {batch_idx}, "
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
        
        accelerator.log({"training_loss": loss, "epoch_num": epoch})

        # 多卡训练时只在主进程打印信息，防止同一份信息打印好多遍
        if accelerator.is_local_main_process:
            progress_bar.update(1) # 更新进度条
            if batch_idx % 200 == 0: # 保留原有的每200个batch打印一次loss的逻辑
                 # 可以在进度条后附加信息
                progress_bar.set_postfix_str(f"Loss: {loss.item():.6f}")
                # 或者直接打印
                # print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    if accelerator.is_local_main_process:
        progress_bar.close() # 关闭进度条

# # 测试函数
# def test_old(dataloader, model, loss_fn, accelerator, device):
#     # eval model
#     model.eval()

#     test_loss = 0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for inputs, targets in dataloader:
            
#             # Compute prediction error
#             outputs = model(inputs)
#             loss = loss_fn(outputs, targets)
            
#             # 计算当前GPU上局部批次的损失总和
#             local_batch_size = targets.size(0)
#             local_sum_loss = loss * local_batch_size
            
#             # 收集所有GPU上的局部损失总和，并加起来得到全局批次的损失总和，累加到test_loss
#             test_loss += accelerator.gather(local_sum_loss).sum().item()
#             pred = outputs.argmax(dim=1, keepdim=True)
            
#             correct += accelerator.gather(pred.eq(targets.view_as(pred)).sum()).sum().item()
#             # ====================================
            
#             total += targets.size(0) * accelerator.num_processes  # 总数可以直接累加
            
#     test_loss /= total
#     accuracy = 100. * correct / total
    
#     if accelerator.is_local_main_process:
#         print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)')

# 测试函数
def test(dataloader, model, loss_fn, accelerator, device, epoch):
    model.eval()

    # 在本卡上累加的局部量
    local_test_loss_sum = torch.tensor(0.0, device=device)
    local_correct_sum = torch.tensor(0, device=device)
    local_sample_sum  = torch.tensor(0, device=device)

    # 仅在主进程创建测试进度条
    if accelerator.is_local_main_process:
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
            
            if accelerator.is_local_main_process:
                test_progress_bar.update(1) # 更新测试进度条
            
    # 全局聚合：收集所有GPU上的局部损失总和，一次性 gather & sum
    total_test_loss = accelerator.gather(local_test_loss_sum).sum()
    total_correct = accelerator.gather(local_correct_sum).sum()
    total_samples = accelerator.gather(local_sample_sum).sum()

    # 只在主进程计算并打印
    if accelerator.is_main_process:
        avg_loss = (total_test_loss / total_samples).item()
        accuracy = 100. * total_correct.item() / total_samples.item()
        print(f'Test set: Average loss: {avg_loss:.4f}, '
              f'Accuracy: {total_correct}/{total_samples} ({accuracy:.0f}%)')
        
def main():
    parser = argparse.ArgumentParser(description="Train a model with configurable GPU for direct Python execution.")
    parser.add_argument(
        "--gpu_id",
        type=str, # gpu_id: 字符串类型，允许指定单个或多个GPU
        default='3', # default=None: 脚本将不主动修改 CUDA_VISIBLE_DEVICES
        help="Specify the GPU ID(s) to use."
    )
    # args: 解析后的命令行参数对象
    args, _ = parser.parse_known_args()

    # 判断是否在分布式训练环境中运行
    is_launched_by_accelerate = 'LOCAL_RANK' in os.environ

    if not is_launched_by_accelerate:
        print(f"[INFO] Running with 'python train.py', setting CUDA_VISIBLE_DEVICES='{args.gpu_id}'")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    else: # accelerate launch 启动
        print(f"[INFO] Running with 'accelerate launch'. GPU selection will be handled by Accelerate.")

    set_seed(42)
    
    # hyperparameters
    config = {
        "num_epoch": 10, # 训练的总轮数
        "batch_num": 1024, # 每个批次的样本数量
        "learning_rate": 1e-4, # l初始学习率
        "mixed_precision": "fp16", # 混合精度训练配置 ('fp16', 'bf16', or 'no')
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
    
    # 初始化Accelerator
    accelerator = Accelerator()
    
    # 获取当前进程信息
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        # 主进程初始化tracker并创建带tracker的accelerator
        tracker = SwanLabTracker("CIFAR10_TRAING")
        accelerator = Accelerator(
            log_with=tracker,
            mixed_precision=mixed_precision
        )
        # 初始化tracker
        accelerator.init_trackers("CIFAR10_TRAING", config=config)
    else:
        # 非主进程只使用混合精度，不使用tracker
        accelerator = Accelerator(mixed_precision=mixed_precision)
    

    # 初始化Accelerator 只有 MULTI_GPU 能这样初始化
    # 训练可视化 ,experiment_name=exp_name
    # tracker = SwanLabTracker("CIFAR10_TRAING") 
    # accelerator = Accelerator(log_with=tracker)
    # accelerator.init_trackers("CIFAR10_TRAING", config=config)
    
    device = accelerator.device # accelerator自动维护的device
    
    # 加载数据集
    train_dataset = datasets.train_dataset_CIFAR10
    test_dataset = datasets.test_dataset_CIFAR10
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break    
    
    # 加载模型
    model = get_model()
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer_instance = optimizers.get_optimizer(model, optimizer_name, lr) # 使用optimizer_name
    
    # 定义学习率
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer_instance, max_lr=25*lr, epochs=epochs, steps_per_epoch=len(train_dataloader)) # 使用optimizer_instance

    # 使用accelerate包装模型、优化器和数据加载器
    model, optimizer_instance, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer_instance, lr_scheduler, train_dataloader, test_dataloader) # 使用optimizer_instance

    # Get logger 用处不明，似乎没有也能记录训练日志
    logger = get_logger(__name__)

    # 检查模型和数据的设备分配情况
    if batch_info_flag and accelerator.is_local_main_process: # 只在需要详细信息且是主进程时打印
        print(f"Model is on device: {next(model.parameters()).device}")
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            print(f"batch_{batch_idx} inputs are on device: {inputs.device}")
            print(f"batch_{batch_idx} targets are on device: {targets.device}")
            if batch_idx > 2: # 示例：只打印前几个批次
                break

    # 主训练循环
    for epoch_num in range(epochs):
        current_epoch = epoch_num + 1
        
        train(train_dataloader, model, loss_fn, accelerator, device, optimizer_instance, lr_scheduler, current_epoch, batch_info=batch_info_flag) 
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