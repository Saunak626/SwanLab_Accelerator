import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import argparse
from tqdm import tqdm
import numpy as np
from safetensors.torch import load_file

# 导入模型和数据集
from src.models.net import get_model
from src.datasets.base_dataset import datasets_CIFAR10

class ModelInference:
    def __init__(self, model_path, device='cuda'):
        """
        初始化推理器
        Args:
            model_path: 模型权重文件路径
            device: 推理设备
        """
        self.device = device
        self.model = get_model().to(device)
        self.model.eval()
        
        # 根据文件类型加载权重
        self.load_weights(model_path)
        
        # CIFAR10类别名称
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def load_weights(self, model_path):
        """加载模型权重"""
        print(f"正在加载模型权重: {model_path}")
        
        if model_path.endswith('.safetensors'):
            # 加载safetensors格式
            state_dict = load_file(model_path)
            self.model.load_state_dict(state_dict)
            print("成功加载 safetensors 格式权重")
            
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            # 加载PyTorch格式
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict)
            print("成功加载 PyTorch 格式权重")
            
        else:
            raise ValueError(f"不支持的文件格式: {model_path}")
    
    def predict_single(self, image_tensor):
        """
        对单个图像进行预测
        Args:
            image_tensor: 形状为 [C, H, W] 的图像张量
        Returns:
            predicted_class, confidence, all_probabilities
        """
        with torch.no_grad():
            # 添加batch维度
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            
            # 前向传播
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # 获取预测结果
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, probabilities.cpu().numpy()[0]
    
    def predict_batch(self, dataloader, max_samples=None):
        """
        批量预测
        Args:
            dataloader: 数据加载器
            max_samples: 最大预测样本数，None表示预测全部
        Returns:
            predictions, ground_truths, accuracies
        """
        predictions = []
        ground_truths = []
        confidences = []
        
        total_samples = 0
        correct_predictions = 0
        
        print("开始批量推理...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="推理进度")):
                if max_samples and total_samples >= max_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                # 获取预测结果
                confidence_scores, predicted = torch.max(probabilities, 1)
                
                # 统计结果
                predictions.extend(predicted.cpu().numpy())
                ground_truths.extend(labels.cpu().numpy())
                confidences.extend(confidence_scores.cpu().numpy())
                
                # 计算准确率
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples
        
        return {
            'predictions': np.array(predictions),
            'ground_truths': np.array(ground_truths),
            'confidences': np.array(confidences),
            'accuracy': accuracy,
            'total_samples': total_samples
        }
    
    def analyze_results(self, results):
        """分析推理结果"""
        predictions = results['predictions']
        ground_truths = results['ground_truths']
        confidences = results['confidences']
        
        print(f"\n=== 推理结果分析 ===")
        print(f"总样本数: {results['total_samples']}")
        print(f"整体准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        
        # 按类别统计准确率
        print(f"\n=== 各类别准确率 ===")
        for class_idx in range(len(self.class_names)):
            class_mask = ground_truths == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predictions[class_mask] == ground_truths[class_mask])
                class_count = np.sum(class_mask)
                print(f"{self.class_names[class_idx]:>12}: {class_accuracy:.4f} ({class_count:>4} 样本)")
        
        # 置信度分析
        print(f"\n=== 置信度分析 ===")
        correct_mask = predictions == ground_truths
        correct_confidence = np.mean(confidences[correct_mask])
        wrong_confidence = np.mean(confidences[~correct_mask])
        print(f"正确预测平均置信度: {correct_confidence:.4f}")
        print(f"错误预测平均置信度: {wrong_confidence:.4f}")

def main():
    parser = argparse.ArgumentParser(description="CIFAR10模型推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型权重文件路径 (.safetensors 或 .pt/.pth)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="数据集路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="批处理大小"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大推理样本数，None表示推理全部数据"
    )
    parser.add_argument(
        "--subset_indices",
        type=str,
        default=None,
        help="指定推理的样本索引，格式: '0,1,2,3' 或 '0-100'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default="./inference_result/results.npz",
        help="保存推理结果的文件路径"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 初始化推理器
    print(f"使用设备: {args.device}")
    inferencer = ModelInference(args.model_path, args.device)
    
    # 加载数据集
    print("加载数据集...")
    _, test_dataset = datasets_CIFAR10(root=args.data_path)
    
    # 处理数据子集
    if args.subset_indices:
        if '-' in args.subset_indices:
            # 范围格式: "0-100"
            start, end = map(int, args.subset_indices.split('-'))
            indices = list(range(start, min(end + 1, len(test_dataset))))
        else:
            # 列表格式: "0,1,2,3"
            indices = [int(x.strip()) for x in args.subset_indices.split(',')]
        
        test_dataset = Subset(test_dataset, indices)
        print(f"使用数据子集，样本数: {len(indices)}")
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 执行推理
    results = inferencer.predict_batch(test_dataloader, args.max_samples)
    
    # 分析结果
    inferencer.analyze_results(results)
    
    # 保存结果
    if args.save_results:
        np.savez(
            args.save_results,
            predictions=results['predictions'],
            ground_truths=results['ground_truths'],
            confidences=results['confidences'],
            accuracy=results['accuracy'],
            class_names=inferencer.class_names
        )
        print(f"\n推理结果已保存到: {args.save_results}")

if __name__ == "__main__":
    main() 