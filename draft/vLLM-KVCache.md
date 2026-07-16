vLLM 中 KVCache Block 的调度与管理

<!-- TODO: 链接 其他文章 -->
KVCache 的调度与管理是 vLLM 中非常核心的特性，我们在其他的文章中分析了 vLLM 中 HMA 的 KVCache 管理机制，当时主要从 KVCache 的分组规划和 buffer 的分组复用角度去窥探一二。现在，我们把视角重新放到调度的一开始，看看在运行时，KVCache 是如何被分配给不同请求或不同 KVCache 组的。

### KVCache 相关数据结构

#### KVCacheBlock

定义：KVCache Block 的所有组成元素


| member values | description | 
|---------------|-------------|
| `block_id` | block 的 index，序号 |
| `ref_cnt` | block 的引用计数，用来标记该 block 被多少个请求占用（因为存在 prefix cache，所以会有多个请求占用同一个 block 的情况）|
| `_block_hash` | block 的 hash 值，为 None 或 带有 group id 的 sha256 bytes 值，仅在 block 存满时计算 |
| `_block_hash_num_tokens` | block 内被 `_block_hash` cover 的 token 数量，用于部分命中 |
| `prev_free_block` | 前序空闲 block |
| `next_free_block` | 后序空闲 block |
| `is_null` | 标志当前 block 是否为 null block 的布尔值 |


| property | description | 
|----------|-------------|
| `block_hash` | `return self._block_hash` |
| `block_hash_num_tokens` | `return self._block_hash_num_tokens` |


| member fuctions | description | input args |
|-----------------|-------------|------------|
| `set_block_hash` | 设置 block hash | block_hash, num_tokens |
| `reset_hash` | 当 block 被驱逐时，重置 block hash 为 None | None |


#### FreeKVCacheBlockQueue

定义：空闲 block 双向队列。一开始这个队列按照 block id 增长的顺序排列，随着 block 的消费和驱逐，block 会按照被驱逐的顺序被重新添加到 free 队列里面：
1. 最近最少使用的 blocks 放在队首 -- LRU
2. 如果两个数据块的最后访问时间相同（由同一请求分配），则拥有更多哈希令牌（即处于 free 队列末端）的那个数据块排在前面。

| member fuctions | description | input args | output |
|-----------------|-------------|------------|--------|
| `popleft` | 弹出队首的第一个 block | None | 第一个队首的 block |
| `popleft_n` | 弹出队首的前 n 个 block | 需要弹出的 block 数量 n | 队首前 n 个 block 的列表 |
| `remove` | 从 free 队列中删除 block | 要删除的 block | / |
| `append` | 向 free 队列中添加 block，放到队尾 | 要添加的 block | / |
| `append_n` | 向 free 队列中添加 n 个 block，放到队尾 | 要添加的 blocks | / |
| `prepend_n` | 向 free 队首添加 n 个 block | 要添加的 block | / |
| `get_all_free_blocks` | 获取所有的 free blocks | / | free blocks list |
| `iter_blocks_after` | 在 cursor 之后，按驱逐顺序遍历 free blocks | cursor block | / |



KVCacheManager 

