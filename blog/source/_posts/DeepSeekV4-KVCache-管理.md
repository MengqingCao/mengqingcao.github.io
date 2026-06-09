---
title: DeepSeek V4 KVCache 管理
date: 2026-06-09 18:00:00
slug: deepseek-v4-kvcache-management
summary: 重点分析了 DeepSeek V4 模型在 kvcache 方面的创新点和 vLLM 及 vLLM Ascend 中是如何实现相应的 KVCache 管理的。
tags: vLLM; Inference; KVCache
categories: vLLM; KVCache; DeepSeek
---

## DeepSeek V4 中的 kvcache 创新

首先，让我们从 DeepSeek V4 官方的技术博客中了解这个模型在 kvcache 这里做了哪些创新。

### 整体架构
从模型整体结构上看，DeepSeek V4 在之前的 attention + moe 的架构中，对残差连接做了一处创新，即引入了mHC。通俗地理解，即在 attention 和 moe 计算之前 先通过 hc_pre 将 hidden states 降维，降维之后的 hidden states 参与 attention 或 moe 计算之后，会通过 hc_post 进行升维，与残差线上的 hidden states concat 到一起作为下一层的 hidden states 输入。这种 mHC 连接使得在尽可能保留更多之前层的信息的同时，对主体 attention 和 moe 的计算量并没有增加。

![image](/img/deepseek_v4/dsv4_arch.png)


### Attention

DeepSeek V4 在 Attention 模块这里引入了较为大刀阔斧的改革，传统的 MLA 对所有 token 的 kv 都进行运算，而 DeepSeek V4 中引入了 kv 压缩的概念，通过一个 compressor 模块对 kvcache 进行压缩，并且设计了两种压缩比：4 和 128，这两种压缩比的 attention 层交替出现；另外 DeepSeek V4 的 mtp 层没有进行压缩，且 Flash 中的第0和1层没有进行压缩。这仅仅是整体上 DeepSeek V4 attention 层的变化，下面让我们进入 c4 和 c128 的 attention 内部探究 kvcache 的更多创新。

#### CSA

首先从最复杂的 c4 层看起，c128的算法设计，除了压缩比的变化，可以认为是 c4 减去 稀疏化的 kv topk select。因此，看懂 c4 之后，c128 自然也就融会贯通了。

c4 这层的 attention 被 DeepSeek V4 命名为 CSA （ Compressed Sparse Attention），其中，所有 kv tokens 的 hidden states 被分为三股：

1. **sliding window kv entries**：固定窗口范围内的 kv entries
2. **compressed kv entries**：经过一个 token 层级的 compressor，将原始的所有 kv 按照指定的压缩比进行压缩。compressor 是一个包含可训练参数的模块，通过对 kv entries 进行 compressor 前向计算后得到 压缩后的 kv entries。但是请注意，这里得到压缩后的 kv entries 并不是直接参与 attention 计算的 kv。
3. **indexer 模块的 kv entries**：与 2 相似地，indexer 模块中，使用 compressor 对输入的 kv 进行压缩，得到压缩后的 indexer keys 与低秩 queries 通过 MQA 计算得到 indexer scores。这与 DeepSeek V3.2 的 DSA 操作是相同的，只是此时选出的 indices 是针对压缩后的 kv entries 的。

这时，我们再回到2，在 2 中得到的压缩后的 kv entries 中根据 indexer 输出的 indices 选择出 selected compressed kv entries 。selected compressed kv entries 与 sliding window 的 kv entries concat 到一起，即为最终 MQA 计算需要的 kv。

![image](/img/deepseek_v4/csa.png)

至此，我们可以发现，需要框架侧去管理的 kvcache 已经有 swa cache、compressed kv cache、compressed indexer kv cache 三种。但 compressor 为了持续实现 kv 的压缩，在不满压缩比的时候，需要把当前的 kv entries 及 kv 压缩过程中的加权和计算时的 softmax 结果，即 score 暂存起来，待 满压缩比 之后进行压缩，才会存入 compressed kv cache 内。因此，每个 compressor 也意味着需要两个 中间暂存的 kv state 和 score state 的 cache。

#### HCA

c128 这层的 attention 被 DeepSeek V4 命名为 HCA （  Heavily Compressed Attentio），他仅仅保留了 sliding window 和 主 compressor 的设计，没有进行 lightning indexer 的稀疏化选择。

![image](/img/deepseek_v4/hca.png)

#### sliding window mla

而对于不进行压缩的层，就只剩下了 sliding window kv 这一种。

总结一下所有层出现的 kv cache：

```
c1: swa cache
c4: swa cache、compressed kv cache、compressed indexer kv cache、c4a kv state，c4a score state、c4i kv state、c4i score state   
c128: swa cache、compressed kv cache、c128a kv state，c128a score state
```

> c4a 指 c4 这一层的 attention，c4i 指 c4 这一层的 indexer；c128 同理

## KVCache planning

如果你对 vLLM 中 kvcache 的分组管理不太了解，这里可以先移步 [vLLM 中的几种 KVCache 分组机制](/posts/vllm-kvcache-grouping/) 了解什么是 `UniformTypeKVCacheSpecs` 和 Hybrid Memory Allocator，DeepSeek V4 就是基于这两种机制进行 kvcache 规划的。


### in vLLM

首先了解 vLLM 在 GPU 上 kv cache planning 的设计：
总体可以概括为，使用 `UniformTypeKVCacheSpecs` 对 page_size 差异大的 specs 进行一层 warp，然后再多个 `UniformTypeKVCacheSpecs` 组之间做 hybrid kv 的共享。
而 `UniformTypeKVCacheSpecs` 组内部，则划分为 3 个大小不一的 tensor，这既能实现组之间 对于同一 buffer 的 block pool 共享，又能防止 HMA 本身需要的 `page_size` 对齐操作引入了过多的 wasted pad。

![image](/img/deepseek_v4/gpu_kv_planning.png)

> 需要注意的一点是，我们可以看到 group 1 的 `block_size` 是 256，但实际在每个 spec 上的 `block_size` 都不相等。这是由于压缩比的存在，我们实际存储 kvcache 的时候不会把所有 token 的 kv cache 存下来，vLLM 把压缩发生在了 `block_size` 这个维度。这也就是说，虽然对于管理面的 `KVCacheManager` 来说看到的 `block_size` 是 256，但实际送到 kernel 侧的 block 已经被压缩为 `block_size // compress_ratio` 个 slots 了。
>

从运行时的视角分析上图，假设当前为 DeepSeek V4 Flash 模型，如下所示为其每层的结构：
> 43 层：
> 0/1: swa
> 2～42 的偶数层：c4
> 2～42 的奇数层：c128

此时 kvcache 分组的组大小为 22，即有 22 * 3 个 `KVCacheTensor`，每个 tensor 被所有 5 个组共享。
对于一个长度为 258 的 `请求，KVCacheManager` 和 `KVCacheCoordinator` 按照不同组的 `SingleTypeKVCacheManager` 的规则给不同的组分配 block 的结果如下：

```
group 1:   cdiv(258, 256) = 2
group 2/3: cdiv(258, 256) = 2
group 4:   cdiv(258, 8) = 33
group 5:   cdiv(258, 4) = 65
```

虽然看起来给 group 4 和 5 分配了非常多的 block，但实际上由于他们用的是 `SlidingWindowManager`，前面的 32 个（group5是64个）blocks 会立即调用 `blockpool.free_blocks` 释放掉，因此这里实际占用的 block 数量只有 1 个。

管理面完成了 block 分配后，经由 ModelRunner 计算 `block_table`、`slot_mapping` 等，就进入 attention 后端执行计算了。我们前面提到，对 `block_size` 进行了压缩，因此 `slot_mapping` 这里实际上也要计算压缩后的 `slot_mapping`。具体的计算逻辑在 `vllm/v1/attention/backends/mla/compressor_utils.py` 中的 `get_compressed_slot_mapping` 方法中，简要概括他主要的行为是，根据实际存储的 `block_size`（即压缩后的）去计算 `slot_mapping`，然后把不满压缩比位置的 `slot_mapping` 值置为 -`1，reshape_and_cache` 算子对 `slot_id` 为 -1 的位置不会做任何操作，这样就实现了实际在 attention 后端的压缩的 kvcache 的存储。

### in vLLM Ascend

在了解 vLLM Ascend 的实现之前，我们首先需要了解 ascend 侧算子的一个限制，这也是我们和 vLLM GPU 方案有区别的原因：attention 算子支持的最小 `block_size` 为 16，性能原因推荐 `block_size` 为 128。

> 需要补充一点：当前正在做 不同 `block_size` 的实验，用来 trade off 不同 `block_size` 对 kvcache 管理造成的 prefix cache hit rate 降低 和 算子性能劣化 带来的性能影响，未来这个方案图也许会发生变化。

![image](/img/deepseek_v4/npu_kv_planning.png)

okay，了解完这个限制之后，比较显然的就是，c128a 的大小比 GPU 的大了很多倍，已经不适合再单独一个 `KVCacheTensor` 去存了，否则将会引入巨大的 patch。即便 `SlidingWindowManager` 会很快释放掉这些 pad blocks，这依然很大提高了临时的 blocks 占用。
解决这个问题一个很直觉的做法是，直接让 c128a 也成为和 其他组去共享 tensor 的一个组，也就是上图所示。这样，在仅仅引入一个组的情况下，即可消除每个组去申请很多的 pad blocks。缺点也很显然，我们要为此实现和 vLLM 略有不同的 KVCache Planning。

看到这里也许你会有一个疑问：vLLM Ascend 的 kvcache 没有压缩吗？为什么图中标的具体 spec 的 `block_size` 都是和管理面一样的？

这就是我们的第二个定制化修改：**压缩 `num_blocks` 这一维，而不是 `block_size` 这一维**。

具体的实现方法是，当 scheduler 为 `num_tokens` 个 tokens 去申请 blocks 时，我们会提前将 num_tokens 压缩为 `num_tokens // compress_ratio` ，这样我们即可在分配 block 的时候，为 compressed kvcache 只分配 1/`compress_ratio` 倍的 blocks，这样 kvcache 的实际压缩就完成了。
不过，虽然成功在 `num_blocks` 这一维完成了压缩，但这个带来的影响是 `block_id` 已经被压缩，因此我们需要在 `ModelRunner` 侧适配 压缩 position id 以及根据压缩后的 position id 去计算 压缩后的 slot_mapping 的逻辑，具体代码可以参考：`vllm_ascend.utils.get_compressed_pos_and_indices` 和 `vllm_ascend.worker.block_table.MultiGroupBlockTable.compute_slot_mapping` 。

运行时的逻辑实际上和 vLLM GPU 上的逻辑基本一致，这里不做赘述。



到这里，DeepSeek V4 的基础 KVCache 适配就讲完了，至于 prefix cache 的优化、KVCache Planning 优化等，vLLM 和 vLLM Ascend 都持续在做更多的工作，未完待续...

## references
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- https://vllm.ai/blog/2026-04-24-deepseek-v4
