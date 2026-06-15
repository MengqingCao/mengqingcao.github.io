标准的逻辑 KVCache Shape：
```
[num_layers_slots, num_blocks, num_heads, num_states, <state_content>]
```
> num_states: 每一个 block 的 token positions，对于 recurrent state 是只有一个
> num_heads: 头数
> <state_content>: attention 后端特定的，总是连续的

对于各种 attention backend 来说，他们的 shape 如下：

| backend | num_states | num_heads | state_content |
| :--- | :---: | ---: | ---: |
| GQA | `block_size` | `num_kv_heads` | `[2, head_dim]` |
| DeepSeekV4 CSA | `block_size/4` | 1 | `[latent_size]` |
| MLA | `block_size` | 1 | `[latent_size]` |
| Mamba2 Conv | 1 | 1 | `[conv_dim/tp, kernel-1]` |
| Mamba2 SSM | 1 | `num_heads` | `[head_dim, state_size]` |

语义 shape 一直是 `[layer_slots, num_blocks, num_heads, num_states, state_content]`
对应简写：`L, B, H, N, C`

而物理 layout 被 stride_order 来控制：

| physical layout | stride order | 优点 | 适用场景 |
| :--- | :---: | ---: | ---: |
| [L, B, H, N, C] | `(0, 1, 2, 3, 4)` | H 维切 TP 之后，同一个 head 对应的 cache 物理内存是连续的 | pd 分离中 tp 不对等 |
| [L, B, N, H, C] | `(0, 1, 3, 2, 4)` | 一种 cache 的所有 cache 物理内存是连续的 | 适用于 block 级别的 kv 传输，pd 分离中异质 block-size |
| [B, L, H, N, C] | `(1, 0, 2, 3, 4)` | 一个 block 的所有层的 cache 物理内存是连续的，connector 可以在一次 RDMA 读取中传输所有层的一个完整 block | 当 tp 和 block-size 都相等时，p/d 传输量也比较大的场景  |
| [B, H, L, N, C] | `(1, 2, 0, 3, 4)` | 一个 block 和 head 对应的所有层的 cache 是连续的 | 异质 tp + 大的 p/d 传输量 |

共享 tensor 对 kv layout 的约束：
共享同一个 kv tensor 的不同 kvcache 类型一般有不同的 H, N, C，这种情况下：
* per-layer-per-block 内容 [H, N, C] ，在物理内存上必须是连续的，所以只有 `[L, B, H, N, C]` 和 `[B, L, H, N, C]` 满足约束
* 如果 `shared_by` 的所有层的 H，N 和 C 的大小都是 match 的，那所有的 layout ，包括 `[B, H, L, N, C]` 和 `[L, B, N, H, C]` 都是合法的。
