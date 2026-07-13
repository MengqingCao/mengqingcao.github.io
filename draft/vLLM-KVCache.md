vLLM 中 KVCache Block 的调度与管理

<!-- TODO: 链接 其他文章 -->
KVCache 的调度与管理是 vLLM 中非常核心的特性，我们在其他的文章中分析了 vLLM 中 HMA 的 KVCache 管理机制，当时主要从 KVCache 的分组规划和 buffer 的分组复用角度去窥探一二。现在，我们把视角重新放到调度的一开始，看看在运行时，KVCache 是如何被分配给不同请求或不同 KVCache 组的。

### KVCache 相关数据结构

#### KVCacheBlock

定义：KVCache Block 的所有组成元素


| member | description | 
|--------|-------------|
| `block_id` | block 的 index，序号 |
| `ref_cnt` | block 的引用计数，用来标记该 block 被多少个请求占用（因为存在 prefix cache，所以会有多个请求占用同一个 block 的情况）|
| `_block_hash` | block 的 hash 值，为 None 或 带有 group id 的 sha256 bytes 值，仅在 block 存满时计算 |
| `_block_hash_num_tokens` | block 内被 `_block_hash` cover 的 token 数量，用于部分命中 |
| `prev_free_block` | 前序空闲 block |
| `next_free_block` | 后序空闲 block |
| `is_null` | 标志当前 block 是否为 null block 的布尔值 |




KVCacheManager 

