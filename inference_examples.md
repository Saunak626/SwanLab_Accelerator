# 模型推理使用示例

## 1. 权重文件类型和加载方式

### 检查点文件结构

```
result/CIFAR10_20250525_221648/
├── epoch_5_checkpoint/
│   ├── pytorch_model/
│   │   ├── mp_rank_00_model_states.pt          # 完整模型权重
│   │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt  # 优化器状态(GPU0)
│   │   └── zero_pp_rank_1_mp_rank_00_optim_states.pt  # 优化器状态(GPU1)
│   ├── random_states_0.pkl                    # 随机状态(GPU0)
│   ├── random_states_1.pkl                    # 随机状态(GPU1)
│   └── scheduler.bin                           # 学习率调度器状态
└── final_model/
    └── model.safetensors                       # 最终完整模型(推荐)

```

### 权重文件说明

- **`model.safetensors`**: 最终完整模型，推荐用于推理
- **`mp_rank_00_model_states.pt`**: 检查点中的模型权重，可用于推理
- **优化器状态文件**: 仅用于恢复训练，推理时不需要

## 2. 推理脚本使用方法

### 基本用法

#### 使用 safetensors 格式（推荐）

```bash
python inference.py --model_path result/CIFAR10_20250525_221648/final_model/model.safetensors
```

#### 使用检查点中的模型权重

```bash
python inference.py --model_path result/CIFAR10_20250525_221648/epoch_5_checkpoint/pytorch_model/mp_rank_00_model_states.pt
```

### 高级用法

#### 推理指定数量的样本

```bash
python inference.py \
    --model_path result/CIFAR10_20250525_221648/final_model/model.safetensors \
    --max_samples 1000
```

#### 推理指定索引的样本

```bash
# 推理前100个样本
python inference.py \
    --model_path result/CIFAR10_20250525_221648/final_model/model.safetensors \
    --subset_indices "0-99"

# 推理指定的几个样本
python inference.py \
    --model_path result/CIFAR10_20250525_221648/final_model/model.safetensors \
    --subset_indices "0,10,20,30,40"
```

#### 保存推理结果

```bash
python inference.py \
    --model_path result/CIFAR10_20250525_221648/final_model/model.safetensors \
    --save_results inference_results.npz
```

#### 指定数据集路径和设备

```bash
python inference.py \
    --model_path result/CIFAR10_20250525_221648/final_model/model.safetensors \
    --data_path ./data \
    --device cuda \
    --batch_size 256
```

## 3. 输出示例

```
使用设备: cuda
正在加载模型权重: result/CIFAR10_20250525_221648/final_model/model.safetensors
成功加载 safetensors 格式权重
加载数据集...
训练数据集长度: 50000
测试数据集长度: 10000
开始批量推理...
推理进度: 100%|██████████| 79/79 [00:03<00:00, 24.32batch/s]

=== 推理结果分析 ===
总样本数: 10000
整体准确率: 0.8234 (82.34%)
平均置信度: 0.7845

=== 各类别准确率 ===
    airplane: 0.8540 (1000 样本)
  automobile: 0.9120 (1000 样本)
        bird: 0.7230 (1000 样本)
         cat: 0.6890 (1000 样本)
        deer: 0.8010 (1000 样本)
         dog: 0.7650 (1000 样本)
        frog: 0.8890 (1000 样本)
       horse: 0.8670 (1000 样本)
        ship: 0.9010 (1000 样本)
       truck: 0.8330 (1000 样本)

=== 置信度分析 ===
正确预测平均置信度: 0.8234
错误预测平均置信度: 0.6123
```

## 4. 编程方式使用

```python
from inference import ModelInference
import torch

# 初始化推理器
inferencer = ModelInference('result/CIFAR10_XXX/final_model/model.safetensors')

# 单张图像推理
image = torch.randn(3, 32, 32)  # CIFAR10图像尺寸
predicted_class, confidence, probabilities = inferencer.predict_single(image)
print(f"预测类别: {inferencer.class_names[predicted_class]}")
print(f"置信度: {confidence:.4f}")

# 批量推理
from torch.utils.data import DataLoader
from src.datasets.base_dataset import datasets_CIFAR10

_, test_dataset = datasets_CIFAR10()
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

results = inferencer.predict_batch(test_loader, max_samples=1000)
inferencer.analyze_results(results)
```

## 5. 注意事项

1. **推荐使用 `model.safetensors`**: 这是最终的完整模型，加载最简单可靠
2. **检查点权重**: `mp_rank_00_model_states.pt` 也可以用于推理，但需要确保模型结构匹配
3. **优化器状态**: `zero_pp_rank_*_optim_states.pt` 文件仅用于恢复训练，推理时不需要
4. **设备兼容**: 脚本会自动处理 CPU/GPU 设备映射
5. **数据预处理**: 推理时会自动应用与训练时相同的数据预处理
