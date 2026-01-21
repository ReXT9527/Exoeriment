"""
神经元相似度筛选脚本

对于维度为 R^{m×n} 的线性层权重矩阵，单个神经元对应矩阵中的一行 R^{1×n}。
本脚本计算 llama2-7b（基础模型）和 llama2-7b-chat（安全对齐模型）
中对应位置神经元向量之间的余弦相似度，筛选出变化最大的神经元。
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import json
import os


@dataclass
class NeuronInfo:
    """存储神经元信息的数据类"""
    layer_name: str       # 层名称
    neuron_idx: int       # 神经元索引（行索引）
    similarity: float     # 余弦相似度

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "neuron_idx": self.neuron_idx,
            "similarity": self.similarity
        }


def compute_cosine_similarity_per_row(
    weight1: torch.Tensor,
    weight2: torch.Tensor
) -> torch.Tensor:
    """
    计算两个权重矩阵中每一行（神经元）之间的余弦相似度

    Args:
        weight1: 第一个模型的权重矩阵 [m, n]
        weight2: 第二个模型的权重矩阵 [m, n]

    Returns:
        similarities: 每行的余弦相似度 [m]
    """
    assert weight1.shape == weight2.shape, \
        f"权重矩阵形状不匹配: {weight1.shape} vs {weight2.shape}"

    # 将权重展平为2D（如果需要）
    if weight1.dim() > 2:
        weight1 = weight1.view(weight1.size(0), -1)
        weight2 = weight2.view(weight2.size(0), -1)

    # 计算L2范数
    norm1 = torch.norm(weight1, dim=1, keepdim=True)
    norm2 = torch.norm(weight2, dim=1, keepdim=True)

    # 避免除零
    norm1 = torch.clamp(norm1, min=1e-8)
    norm2 = torch.clamp(norm2, min=1e-8)

    # 归一化
    weight1_normalized = weight1 / norm1
    weight2_normalized = weight2 / norm2

    # 计算余弦相似度（逐行点积）
    similarities = (weight1_normalized * weight2_normalized).sum(dim=1)

    return similarities


def get_linear_layer_names(model) -> List[str]:
    """
    获取模型中所有线性层的名称

    Args:
        model: HuggingFace模型

    Returns:
        layer_names: 线性层名称列表
    """
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    return linear_layers


def load_model_weights(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """
    加载模型的线性层权重

    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备
        dtype: 数据类型

    Returns:
        weights: 包含线性层权重的字典
    """
    print(f"正在加载模型: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights[name] = module.weight.data.clone().to(device)

    # 释放模型内存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return weights


def compute_all_neuron_similarities(
    weights1: Dict[str, torch.Tensor],
    weights2: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    计算所有线性层中神经元的余弦相似度

    Args:
        weights1: 第一个模型的权重字典
        weights2: 第二个模型的权重字典

    Returns:
        similarities: 每层神经元相似度的字典
    """
    similarities = {}

    common_layers = set(weights1.keys()) & set(weights2.keys())
    print(f"共有 {len(common_layers)} 个线性层")

    for layer_name in tqdm(common_layers, desc="计算相似度"):
        w1 = weights1[layer_name]
        w2 = weights2[layer_name]

        if w1.shape != w2.shape:
            print(f"警告: 层 {layer_name} 形状不匹配，跳过")
            continue

        sim = compute_cosine_similarity_per_row(w1, w2)
        similarities[layer_name] = sim

    return similarities


def filter_changed_neurons(
    similarities: Dict[str, torch.Tensor],
    top_k: Optional[int] = None,
    threshold: Optional[float] = None
) -> List[NeuronInfo]:
    """
    筛选变化最大的神经元（相似度最低的神经元）

    Args:
        similarities: 每层神经元相似度的字典
        top_k: 返回变化最大的前k个神经元
        threshold: 相似度阈值，低于此值的神经元会被选中

    Returns:
        changed_neurons: 变化最大的神经元列表
    """
    all_neurons = []

    for layer_name, sim in similarities.items():
        for idx in range(sim.size(0)):
            all_neurons.append(NeuronInfo(
                layer_name=layer_name,
                neuron_idx=idx,
                similarity=sim[idx].item()
            ))

    # 按相似度升序排序（相似度越低，变化越大）
    all_neurons.sort(key=lambda x: x.similarity)

    # 根据条件筛选
    if threshold is not None:
        all_neurons = [n for n in all_neurons if n.similarity < threshold]

    if top_k is not None:
        all_neurons = all_neurons[:top_k]

    return all_neurons


def analyze_by_layer(
    similarities: Dict[str, torch.Tensor]
) -> Dict[str, Dict]:
    """
    按层分析神经元变化统计

    Args:
        similarities: 每层神经元相似度的字典

    Returns:
        layer_stats: 每层的统计信息
    """
    layer_stats = {}

    for layer_name, sim in similarities.items():
        sim_np = sim.cpu().numpy()
        layer_stats[layer_name] = {
            "num_neurons": len(sim_np),
            "mean_similarity": float(np.mean(sim_np)),
            "std_similarity": float(np.std(sim_np)),
            "min_similarity": float(np.min(sim_np)),
            "max_similarity": float(np.max(sim_np)),
            "median_similarity": float(np.median(sim_np)),
            "num_below_0.9": int(np.sum(sim_np < 0.9)),
            "num_below_0.8": int(np.sum(sim_np < 0.8)),
            "num_below_0.5": int(np.sum(sim_np < 0.5)),
        }

    return layer_stats


def save_results(
    changed_neurons: List[NeuronInfo],
    layer_stats: Dict[str, Dict],
    output_dir: str = "results"
):
    """
    保存分析结果

    Args:
        changed_neurons: 变化最大的神经元列表
        layer_stats: 层统计信息
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存变化最大的神经元
    neurons_data = [n.to_dict() for n in changed_neurons]
    with open(os.path.join(output_dir, "changed_neurons.json"), "w") as f:
        json.dump(neurons_data, f, indent=2, ensure_ascii=False)

    # 保存层统计信息
    with open(os.path.join(output_dir, "layer_stats.json"), "w") as f:
        json.dump(layer_stats, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到 {output_dir} 目录")


def main(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    chat_model: str = "meta-llama/Llama-2-7b-chat-hf",
    top_k: int = 1000,
    threshold: Optional[float] = None,
    output_dir: str = "results",
    device: str = "cpu"
):
    """
    主函数：计算并筛选变化最大的神经元

    Args:
        base_model: 基础模型路径
        chat_model: Chat对齐后的模型路径
        top_k: 返回变化最大的前k个神经元
        threshold: 相似度阈值
        output_dir: 输出目录
        device: 计算设备
    """
    print("=" * 60)
    print("神经元相似度筛选")
    print("=" * 60)
    print(f"基础模型: {base_model}")
    print(f"Chat模型: {chat_model}")
    print("=" * 60)

    # 加载模型权重
    print("\n[1/4] 加载基础模型权重...")
    weights_base = load_model_weights(base_model, device=device)

    print("\n[2/4] 加载Chat模型权重...")
    weights_chat = load_model_weights(chat_model, device=device)

    # 计算相似度
    print("\n[3/4] 计算神经元余弦相似度...")
    similarities = compute_all_neuron_similarities(weights_base, weights_chat)

    # 分析和筛选
    print("\n[4/4] 分析和筛选变化最大的神经元...")

    # 按层统计
    layer_stats = analyze_by_layer(similarities)

    # 筛选变化最大的神经元
    changed_neurons = filter_changed_neurons(
        similarities,
        top_k=top_k,
        threshold=threshold
    )

    # 输出统计信息
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)

    # 找出变化最大的层
    layers_by_change = sorted(
        layer_stats.items(),
        key=lambda x: x[1]["mean_similarity"]
    )

    print("\n变化最大的5个层（平均相似度最低）:")
    for layer_name, stats in layers_by_change[:5]:
        print(f"  {layer_name}: 平均相似度={stats['mean_similarity']:.4f}, "
              f"最小={stats['min_similarity']:.4f}")

    print(f"\n变化最大的10个神经元（相似度最低）:")
    for i, neuron in enumerate(changed_neurons[:10]):
        print(f"  {i+1}. {neuron.layer_name}[{neuron.neuron_idx}]: "
              f"相似度={neuron.similarity:.4f}")

    # 保存结果
    save_results(changed_neurons, layer_stats, output_dir)

    return changed_neurons, layer_stats, similarities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="神经元相似度筛选")
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="基础模型路径"
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Chat对齐后的模型路径"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="返回变化最大的前k个神经元"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="相似度阈值"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备"
    )

    args = parser.parse_args()

    main(
        base_model=args.base_model,
        chat_model=args.chat_model,
        top_k=args.top_k,
        threshold=args.threshold,
        output_dir=args.output_dir,
        device=args.device
    )
