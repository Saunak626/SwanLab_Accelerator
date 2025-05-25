# 通用深度学习训练框架 (基于 Accelerate 和 SwanLab)

本项目是一个基于 Hugging Face Accelerate 和 SwanLab 构建的通用深度学习训练框架模板，旨在提高实验效率、代码复用性和结果可复现性。

## 主要特性

- **模块化设计**: 模型、数据集、损失函数、优化器、学习率调度器、训练引擎等核心组件解耦，易于扩展和替换。
- **配置驱动**: 通过 YAML 配置文件控制实验的各个方面，减少代码修改。
- **易用性**: 提供清晰的脚本入口和辅助函数，方便快速上手。
- **分布式训练**: Leveraging Accelerate for seamless single-GPU, multi-GPU, and TPU training.
- **实验跟踪**: 集成 SwanLab (或其他 tracker) 进行实验参数、指标、媒体等的可视化和记录。
- **可复现性**: 通过配置文件和随机种子控制，确保实验结果的可复现性。

## 项目结构

```
my_dl_framework/
├── .gitignore
├── README.md
├── requirements.txt
├── accelerate_config.yaml  # accelerate 默认配置文件 (可选, 通过 `accelerate config` 生成)
│
├── configs/                  # 实验配置文件 (YAML)
│   ├── base_config.yaml      # 基础配置
│   └── cifar10_resnet18.yaml # CIFAR10 + ResNet18 示例配置
│
├── src/                      # 主要源代码
│   ├── datasets/             # 数据集处理
│   │   ├── __init__.py
│   │   ├── base_dataset.py   # 数据集基类
│   │   └── cifar10_dataset.py  # (示例) CIFAR10 数据集实现
│   │
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   ├── base_model.py     # 模型基类
│   │   └── resnet.py         # (示例) ResNet 实现
│   │
│   ├── losses/               # 损失函数
│   │   ├── __init__.py
│   │   └── cross_entropy.py  # (示例)
│   │
│   ├── metrics/              # 评估指标
│   │   ├── __init__.py
│   │   └── accuracy.py       # (示例)
│   │
│   ├── optimizers/           # 优化器
│   │   ├── __init__.py
│   │   └── adamw.py          # (示例)
│   │
│   ├── schedulers/           # 学习率调度器
│   │   ├── __init__.py
│   │   └── one_cycle_lr.py   # (示例)
│   │
│   ├── engine/               # 训练、评估、推理逻辑
│   │   ├── __init__.py
│   │   ├── trainer.py        # 核心训练器类
│   │   └── predictor.py      # (可选) 推理逻辑
│   │   └── evaluator.py      # (可选) 评估逻辑
│   │
│   ├── utils/                # 工具函数
│   │   ├── __init__.py
│   │   ├── config_parser.py  # 解析配置文件
│   │   ├── checkpoint.py     # 模型保存与加载
│   │   └── misc.py           # 其他工具函数 (如设置随机种子)
│   │
│   └── main_setup.py       # 根据配置初始化各组件的辅助函数
│
├── train.py                  # 训练脚本入口
├── infer.py                  # 推理脚本入口
├── evaluate.py               # 评估脚本入口
│
├── scripts/                  # shell 脚本
│   └── train_cifar10.sh      # (示例) CIFAR10 训练启动脚本
│
└── outputs/                  # 训练输出 (模型权重, 日志等)
    └── <project_name>/
        ├── <experiment_name>/
        │   ├── checkpoints/
        │   └── logs/         # (例如 TensorBoard 日志)
        └── swanlog/          # SwanLab 日志 (如果使用)

```

## 环境搭建

1.  **克隆项目**

    ```bash
    git clone <your-repo-url>
    cd my_dl_framework
    ```

2.  **创建并激活 conda 环境 (推荐)**

    ```bash
    conda create -n my_dl_env python=3.9
    conda activate my_dl_env
    ```

3.  **安装依赖**
    首先，根据你的 PyTorch 版本和 CUDA 版本安装 PyTorch。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应的安装命令。
    例如 (CUDA 11.8):

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    然后安装项目特定依赖:

    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` 应包含以下核心依赖 (及其他所需库):

    ```
    accelerate
    omegaconf
    pyyaml
    swanlab
    # torch, torchvision, torchaudio (建议通过官网指令安装)
    # Add other libraries like numpy, scikit-learn, etc. as needed
    ```

4.  **配置 Accelerate (可选, 但推荐用于多设备或特定硬件)**
    ```bash
    accelerate config
    ```
    根据提示配置你的训练环境 (单 GPU, 多 GPU, TPU, 混合精度等)。这会生成一个默认的 `accelerate_config.yaml` 文件。
    你也可以在 `accelerate launch` 命令中通过参数覆盖这些配置。

## 使用说明

所有操作都通过根目录下的脚本 (`train.py`, `infer.py`, `evaluate.py`) 启动，并由 YAML 配置文件驱动。

### 1. 配置文件

- `configs/base_config.yaml`: 包含所有实验通用的基础配置。
- `configs/<your_experiment_config>.yaml`: 特定实验的配置，会覆盖 `base_config.yaml` 中的同名设置。OmegaConf 用于加载和合并配置。

### 2. 训练模型

通过 `train.py` 脚本启动训练。你需要指定一个配置文件。

**直接使用 `accelerate launch` (推荐):**

```bash
accelerate launch train.py --config_file configs/cifar10_resnet18.yaml
```

这条命令会使用 `accelerate` 来启动训练，自动处理设备分配。

**通过 Shell 脚本 (封装了 `accelerate launch`):**
项目提供了一个示例脚本 `scripts/train_cifar10.sh`。

```bash
bash scripts/train_cifar10.sh configs/cifar10_resnet18.yaml
```

如果不想传递配置文件路径，脚本会使用默认的 `configs/cifar10_resnet18.yaml`。
你可以在脚本内部修改 `accelerate launch` 的参数，或者直接在命令行覆盖配置参数：

```bash
accelerate launch train.py --config_file configs/cifar10_resnet18.yaml optimizer.params.lr=0.0005 training.batch_size=256
```

### 3. 模型评估

使用 `evaluate.py` 脚本评估已训练模型的性能。

```bash
accelerate launch evaluate.py \
    --config_file configs/cifar10_resnet18.yaml \
    --checkpoint_path outputs/CIFAR10_Experiments/swanlog/<experiment_run_id>/checkpoints/epoch_xx # 替换为你的检查点路径
```

- `--config_file`: 用于加载模型结构、数据预处理等评估所需的配置。
- `--checkpoint_path`: 指向训练好的模型检查点文件或目录 (根据 `accelerator.save_state` 的保存方式)。

### 4. 模型推理

使用 `infer.py` 脚本对新数据或特定样本进行预测。

```bash
accelerate launch infer.py \
    --config_file configs/cifar10_resnet18.yaml \
    --checkpoint_path outputs/CIFAR10_Experiments/swanlog/<experiment_run_id>/checkpoints/epoch_xx \
    # --sample_ids "0,1,2,3" # (可选) 指定从测试集中抽取的样本ID进行可视化
    # --input_path /path/to/your/image.jpg # (可选) 指定单个图片路径进行推理
```

- `--config_file`: 用于加载模型结构和预处理。
- `--checkpoint_path`: 指向训练好的模型检查点。
- 推理脚本的行为 (例如，是对整个数据集、特定 ID 样本还是单个文件进行推理) 需要在 `infer.py` 中具体实现。
- 如果配置了 SwanLab，推理过程中的可视化结果（如图像和预测标签）可以被记录。

### 5. SwanLab 集成与可视化

- **初始化**: 在 `train.py` (或其他脚本) 中，如果配置文件中 `project.tracker` 设置为 `"swanlab"`，则会初始化 `SwanLabTracker`。

  ```python
  # In train.py (simplified)
  # if config.project.get('tracker') == 'swanlab':
  #     swanlab_tracker = SwanLabTracker(
  #         project_name=config.project.name,
  #         experiment_name=config.swanlab.get('experiment_name'), # or auto-generated
  #         config=OmegaConf.to_container(config, resolve=True) # Log hyperparams
  #     )
  #     accelerator = Accelerator(log_with=swanlab_tracker, ...)
  # else:
  #     accelerator = Accelerator(...)

  # accelerator.init_trackers(project_name=config.project.name, config=...)
  ```

- **记录**: 使用 `accelerator.log({"metric_name": value}, step=global_step)` 记录标量指标。
  对于图像等媒体数据，使用 `swanlab.Image()` 等对象并通过 `accelerator.log()` 记录。
- **查看**: 训练开始后，SwanLab 通常会自动启动一个本地服务 (如 `http://localhost:6092`)。
  你也可以手动启动：
  ```bash
  swanlab watch
  ```
  或者指定日志目录：
  ```bash
  swanlab watch -l ./outputs/<project_name>/swanlog # 根据实际 SwanLab 日志路径调整
  ```
  远程访问 SwanLab UI 可能需要内网穿透或配置服务器防火墙。

## 如何扩展

1.  **添加新的数据集**:

    - 在 `src/datasets/` 下创建新的数据集类 (继承 `BaseDataset`)。
    - 在 `src/datasets/__init__.py` 中的 `get_dataset` 工厂函数中注册你的数据集。
    - 在配置文件中指定新的 `dataset.name` 和相关参数。

2.  **添加新的模型**:

    - 在 `src/models/` 下创建新的模型类 (继承 `BaseModel`)。
    - 在 `src/models/__init__.py` 中的 `get_model` 工厂函数中注册。
    - 在配置文件中指定新的 `model.name` 和相关参数。

3.  **添加新的损失/优化器/调度器/指标**:

    - 在对应模块 (`src/losses/`, `src/optimizers/`, `src/schedulers/`, `src/metrics/`) 下创建实现。
    - 在各自的 `__init__.py` 工厂函数中注册。
    - 在配置文件中指定名称和参数。

4.  **自定义训练逻辑**:
    - 修改 `src/engine/trainer.py` 中的 `Trainer` 类。
    - 对于完全不同的训练范式，可以创建新的训练引擎类。

## 注意事项

- 确保所有自定义模块（模型、数据集等）的工厂函数 (`get_model`, `get_dataset` 等）在相应的 `__init__.py` 文件中被正确实现和导入，目前这些文件中的工厂函数多为注释掉的占位符。
- `src/main_setup.py` 中的 `setup_components` 函数依赖于这些工厂函数来实例化组件，需要取消注释并完善其实现，特别是数据加载和模型初始化的部分。
- `train.py`, `infer.py`, `evaluate.py` 中的组件实例化和准备逻辑也依赖于 `setup_components` 的完善。
- 具体的 `cifar10_dataset.py` 和 `resnet.py` 等示例实现需要用户根据实际情况添加。
- `requirements.txt` 文件需要被创建并包含所有必要的依赖。

```

```
