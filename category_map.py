"""
统一分类映射表：把全部故障条目按编号前缀映射到按故障领域的一维 category。
数字前缀的条目（1.1-15.x）保持原有 15 类不变（2026-09-14 起含 14 条早期实测条目）。
K* 前缀的 111 条（KV 缓存扩展）按 locus 内容归入对应领域（大部分是 KV 缓存，少数归解码/采样或分布式）。
"""

# K* 前缀条目：按 locus 归类
# 大部分 locus 都是 KV 缓存/注意力 领域，少数例外：
#   cross-node-transfer → 分布式/多GPU（P/D 分离式、RDMA、跨节点）
#   control-flow → 解码/采样（推测解码、beam search、CUDA graph）
L2_LOCUS_CATEGORY_MAP = {
    "value-corruption": "KV缓存/注意力",
    "index-position": "KV缓存/注意力",
    "block-slot-alloc": "KV缓存/注意力",
    "cross-request": "KV缓存/注意力",
    "cross-node-transfer": "分布式/多GPU",
    "eviction-compression": "KV缓存/注意力",
    "arch-specific-cache": "KV缓存/注意力",
    "control-flow": "解码/采样",
    "GLM+Kimi来源补充": "KV缓存/注意力",
    "框架/卸载/共享KV/布局/API缓存": "KV缓存/注意力",
}

# 统一的类别顺序（按条目数从多到少）
CATEGORY_ORDER = [
    "KV缓存/注意力",
    "解码/采样",
    "量化",
    "模型图/架构",
    "服务栈/缓存",
    "硬件/SDC",
    "幻觉/对齐",
    "分布式/多GPU",
    "比特翻转攻击",
    "位置编码/RoPE",
    "长上下文能力",
    "权重/加载",
    "分词器/模板",
    "数值/精度",
    "GPU显存",
]
