---
title: vLLM 中的几种 KVCache 分组机制
date: 2026-06-09 14:34:51
slug: vllm-kvcache-grouping
summary: 整理了 vLLM 在混合 Attention 模型下几种 KVCache 分组方式，以及 HMA 为什么存在、解决了什么问题。
tags: vLLM; Inference
---

KVCache 的分组是 HMA 中非常重要的部分，HMA 是为了解决在混合 Attention 模型下的 KVCache 高效管理诞生的技术。其核心在于，通过分组实现两种 Attention 类型对应的 KVCache 共享同一块 Buffer。

首先统一语言，这一节中，KVCache 类型指不同的 `KVCacheSpec`，大小指 一个 block 占的物理内存字节数，通常为
```
block_size * num_head * head_dim * sizeof(cache_dtype)
```


## 统一的 KVCacheSpec 组

每个 Attention 层的 KVCache 完全一致，完全一致是指，类型一致，大小一致。这是最广泛最基础的 KVCache 组，几乎所有的非混合 Attention 模型均属于这一种组，典型的模型如 Qwen2.5，DeepSeek V3.1等。

这种组中，每一个 attention 层都有一块 kvcache buffer，每个 buffer 上都有 `num_gpu_blocks` 个 blocks，由 `KVCacheManager` 统一从 `BlockPool` 调度即可。

![image](/img/kvcache_grouping/uniform_kvcache.png)


## 类型统一的 KVCacheSpec 组

不同 Attention 层，KVCache type 相同，但大小不一样，典型的模型有 DeepSeek-V3.2（indexer cache层和 MLA 层量化策略不一致）、minicpm 4.1、spec decode 或 MTP 层中的 kv head 数量 和 主模型 不一致的
> 特性 pr：https://github.com/vllm-project/vllm/pull/25101

这种组中，同样是 每一个 attention 层都有一块 kvcache buffer，每个 buffer 上都有 `num_gpu_blocks` 个 blocks，只是每层的 buffer 的大小不一致。因此在这种分组策略下，同样同一个 token 在不同层上 分配的 block id 是相同的。

![image](/img/kvcache_grouping/uniform_type_kvcache.png)

## 混合 KVCacheSpec 组

不同 Attention 层，KVCache type 不同，大小不一定一样。
典型模型有 Gemma3、Qwen3-Next、Qwen3.5等。

这里就是 HMA 的核心内容了。

### 首先理解为什么有这个问题。

让我们站在 Qwen3-Next 没有 HMA 的时间点上，此时 vLLM 的行为是，给每层分配同样大小的一块 buffer。（ps：可能有的同学会疑惑，这不是刚好可以用 `UniformTypeSpecs`？第一点是 vLLM 中 HMA 的出现早于 `UniformTypeSpecs`。另外还有更重要的一点，我们放在后面解释）

每层分配同样大小的 buffer，看到这里，结合 linear attention 和 full attention 的 kvcache 的区别，大家应该看到了问题，是的，对于 linear attention 层来说，这是极大的浪费。因为对于每个请求来说，linear attention 需要的 kvcache 的总大小是固定的，也就是说，只需要固定 `N` 个 block（N=1），但是还是强行给他分配了 `NUM_BLOCKS` 个 block，并且在  Qwen3-Next 中，linear attention 层实际上比 full attention 层还要多 2 倍，这无疑带来了更大的 kvcache 浪费。

那么，如何解决这个问题呢？让我们首先尝试用 `UniformKVCacheSpecs` 的思路解决问题，这里主要是解释，为什么 `UniformKVCacheSpecs` 并不是最佳方案，跳过也完全不影响对 vLLM 当前 KVCache 方案的理解。

### UniformKVCacheSpecs 方案分析

`UniformKVCacheSpecs` 的设计在一定程度上其实比较契合上述问题的背景，既然某些层需要的 kvcache 比较小，那我直接按照你的需求，给每一层分配不同大小的 kvcache 不就可以了？

一个问题，`UniformKVCacheSpecs` 机制实际上还是要求每一层的 block 数量相等，因为 vLLM 的 `KVCacheCoordinator` 中只有一个 block pool，而且当只有一组 kvcache 时，只有一个 `SingleTypeKVCacheManager`，这意味着对于一个请求来说，在每一层分配的 block id 是相同的，那么我们想要的，给 linear attention 层只分配一个block就无法做到。

其实也不是完全不可以做到，给 linear attention 层只分配一个 block，但是这需要上层的 `KVCacheCoordinator` 支持多 block pool，从两个 block pool 中分别给 linear attention 层和 full attention 层分配 block ，这带来的不仅仅是代码架构的更改，更重要的是我们要给 linear attention 层预留多少个 block 的问题。我们前面所说的只给 linear attention 层分配一个 block，是没有考虑 prefix cache 问题的，当考虑 prefix cache 时， 无法预知运行时会有多少个 block 被命中，即运行时 linear attention 对 kvcache 的 block 数量需求的峰值是无法预估的。而如果按照最保守的方式去为 linear attention 层预留 block，那么当前方案就会退化为每层的 block 数一样。因此，`UniformKVCacheSpecs` 并不能很好地解决 hybrid model 中 KVCache 浪费的问题。

### HMA

首先建立整体认知，HMA 的核心是，不同层去共享同一块 buffer，一个 block pool，一份 blocktable，从而减少 KVCache 的浪费及碎片化
下图以 Qwen3-Next 为例，直观感受一下 HMA 的样子。普通 full attention 和 linear attention 的层数比例为 1:3，因此 shared_by 中，一层 full attention 层和 3 层 linear attention 层 共享同一块 buffer。达成的效果是，对于一个 token，每个 KVCacheTensor 上会分配 4 个 block id，shared by 中的各层每层一个。

![image](/img/kvcache_grouping/hybrid_kvcache.png)

vLLM 在探索 HMA 中 kvcache 排布也经历了几个微调，本文仅介绍当前的最新代码，想了解更多请阅读 vLLM 团队写的博客：

[hybrid models as first class citizens in vllm](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)

HMA 中最重要的两点：

1. 不同类型的 KVCacheSpec 的 page size （一个 block 的物理内存大小）要对齐。vLLM 中当前有两种对齐的方法：
    1. 放大 block size 为原来的 `x` 倍
    2. 添加一个 pad
2. 不同类型的 kvcache 虽然共享一个 buffer，但都有自己单独的一份内存视图，**即物理内存是同一片，但对物理内存的解读不同**。

HMA 兼容更多 attention backend 的工作：将算子支持 block size 和 KVCacheSpec 的 page size 解耦，具体来讲，attention 算子接收到的 block size、block table、slot mapping等，都是基于算子支持的 block size 去计算的。 而调度侧还是通过大的 block size 去完成 block 的分配等工作，可以理解为，调度侧一下子分配了 x 个 kernel 侧的 blocks。这个特性使得 更多的 attention backend 可以使用 HMA。

[[Hybrid]: Decouple Kernel Block Size from KV Page Size](https://github.com/vllm-project/vllm/pull/24486)
