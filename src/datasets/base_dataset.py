import os
import torch

from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

def datasets_CIFAR10(root='./data'):
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('测试数据集长度: {}'.format(len(test_dataset)))
    
    return train_dataset, test_dataset

def show_pic(dataset): # 展示dataset里的6张图片
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)  # 创建 DataLoader
    examples = enumerate(dataloader)  # 组合成一个索引序列
    batch_idx, (example_data, example_targets) = next(examples)
    
    classes = train_dataset.classes
    
    print(f"加载的批次数据形状: {example_data.shape}")
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        img = example_data[i]  # 假设 example_data 是 [batch, 3, 32, 32]
        if len(img.shape) != 3:  # 检查形状，确保是3D
            print(f"警告: 图像 {i} 形状不正确: {img.shape}")  # 调试输出
            continue  # 跳过不正确的图像
        # 调整为适合 plt.imshow 的形状 (H, W, C)
        img = img.permute(1, 2, 0)  # 直接使用 permute
        plt.imshow(img, interpolation='none')
        plt.title(classes[example_targets[i].item()])
        plt.xticks([])
        plt.yticks([])
        print(f"成功处理图像 {i}，形状: {img.shape}")  # 添加调试打印
    
    # 添加保存逻辑：构建上层目录的路径并保存图片
    current_path = os.path.abspath(".")
    parent_dir = os.path.dirname(os.path.dirname(current_path))
    save_dir = os.path.join(parent_dir, "visualization", "output")
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, "output_image.png")
    plt.savefig(save_path)  # 保存图片
    plt.show()  # 保留原有显示逻辑

if __name__ == "__main__":
    
    # 获取当前路径的绝对路径
    current_path = os.path.abspath(".")

    # 获取上两层目录
    parent_dir = os.path.dirname(os.path.dirname(current_path))

    # 拼接上三层目录下的 data 文件夹路径
    target_dir = os.path.join(parent_dir, "data")
    
    train_dataset, test_dataset = datasets_CIFAR10(target_dir)
    
    print("目标路径:", target_dir)
    print("标签类型:", type(train_dataset.targets[0]))
    print("类别名称:", train_dataset.classes)
    

    show_pic(train_dataset)
    