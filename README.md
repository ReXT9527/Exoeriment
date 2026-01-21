# Exoeriment

## 神经元相似度筛选

基于余弦相似度筛选安全对齐前后变化最大的神经元。

### 原理

对于维度为 $\mathbb{R}^{m \times n}$ 的线性层权重矩阵：
- 单个神经元对应矩阵中的一行，表示为 $\mathbb{R}^{1 \times n}$
- 通过计算 llama2-7b（基础模型）和 llama2-7b-chat（安全对齐模型）中对应位置神经元向量之间的余弦相似度
- 筛选出相似度最低的神经元，即安全对齐后变化最大的神经元

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

```bash
python neuron_similarity_filter.py \
    --base-model meta-llama/Llama-2-7b-hf \
    --chat-model meta-llama/Llama-2-7b-chat-hf \
    --top-k 1000 \
    --output-dir results
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-model` | `meta-llama/Llama-2-7b-hf` | 基础模型路径 |
| `--chat-model` | `meta-llama/Llama-2-7b-chat-hf` | Chat对齐后的模型路径 |
| `--top-k` | 1000 | 返回变化最大的前k个神经元 |
| `--threshold` | None | 相似度阈值，低于此值的神经元被选中 |
| `--output-dir` | `results` | 输出目录 |
| `--device` | `cuda`/`cpu` | 计算设备 |

### 输出文件

- `results/changed_neurons.json`: 变化最大的神经元列表
- `results/layer_stats.json`: 每层的统计信息

### API 使用

```python
from neuron_similarity_filter import (
    load_model_weights,
    compute_all_neuron_similarities,
    filter_changed_neurons,
    analyze_by_layer
)

# 加载权重
weights_base = load_model_weights("meta-llama/Llama-2-7b-hf")
weights_chat = load_model_weights("meta-llama/Llama-2-7b-chat-hf")

# 计算相似度
similarities = compute_all_neuron_similarities(weights_base, weights_chat)

# 筛选变化最大的神经元
changed_neurons = filter_changed_neurons(similarities, top_k=1000)

# 按层分析统计
layer_stats = analyze_by_layer(similarities)
```
