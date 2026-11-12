# nano-vllm 投机解码（ngram 草稿）：设计说明

分支：`yczhou25/nano-vllm` 的 `spec-decode`（基于 `async-scheduling`）。用 `LLM(path, num_speculative_tokens=k)` 开启，默认关闭。暂时不能和 `async_scheduling` 同时开。

## 要解决的问题

decode 每步每条序列只出一个 token。batch 小的时候，GPU 每步的时间主要花在读权重和 KV cache 上，算力大部分空着。如果能先猜出后面几个 token，让模型一次前向把它们都验一遍，猜对几个就一步出几个 token，而且多算的几个位置几乎不花额外时间。

ngram（prompt lookup）是最便宜的猜法：不用草稿模型，直接在已有文本里找和结尾相同的片段，把它后面跟着的内容当草稿。代码修改、摘要、RAG 问答这类输出大段复述输入的任务，命中率很高。

## 做法

```
schedule():   每条 decode 序列：草稿 d1..dk = ngram(序列)，预留 k+1 个位置的块
run():        输入 [t, d1..dk]，位置 L-1..L+k-1，一次前向得到 k+1 行 logits
              拒绝采样：接受前 a 个草稿，再采一个 token，共 a+1 个
postprocess():追加 a+1 个 token（碰到 EOS / max_tokens 就截断），
              num_cached_tokens 回退到 L+a，只给 KV 有效的块做哈希
```

### 关键点

1. **草稿（`nanovllm/engine/ngram.py`）**：每条序列维护一个字典，从 n-gram（n = 4 到 1）映射到它最近一次出现后的位置，随序列增长增量更新。查找时先试长的 n-gram。字典里只放结尾之前的 n-gram，所以不会匹配到结尾自己。如果匹配位置离结尾不到 k 个 token，就按这个周期把重复的模式续下去（`a b a b` 会猜 `a b a b`）。完全找不到时，重复最后一个 token。
2. **验证**：每条序列的输入是 `[最后一个 token, d1..dk]`，所有序列都是 k+1 个 query，形状统一，所以 decode 仍然用 `flash_attn_with_kvcache`（q 是 `(B, k+1, H, D)`，causal 掩码按右下角对齐，`cache_seqlens = L+k`）。CUDA graph 还是按序列数抓图，缓冲区放大到 `max_bs*(k+1)` 行。lm_head 对 decode 的每一行都算 logits。
3. **拒绝采样（`RejectionSampler`）**：ngram 草稿是确定性的，相当于提议分布是 one-hot。按标准的投机采样规则：草稿 di 以概率 p_i(di) 接受；第一次拒绝时，从去掉 di 的 p_i 里采样（这就是 `max(0, p - q)` 归一化后的分布）；全部接受时，从最后一行再采一个 bonus token。这样输出的分布和一个一个采样完全一样。贪心行（temperature 0）只在草稿等于 argmax 时接受，输出和不开投机解码的贪心结果一致。实现上只对 k 行做 logsumexp 这类归约，最后只取每条序列一行 `(B, V)` 来采样，不额外存 `(B*(k+1), V)` 的 float 概率表。
4. **KV 和块的记账**：`commit` 先按 k+1 个 token 推进 `num_cached_tokens`；`postprocess` 知道接受了几个以后，再回退到 `L+a`。被拒绝的草稿留在 KV 里的值是脏的，但它们的位置都在下一步要写的范围里，下一步一定会先覆盖再读。块哈希（prefix cache）只覆盖 KV 有效的位置，所以脏数据不会进 prefix cache。块不用回收：下一步需要的块数一定不少于这一步。
5. **张量并行**：草稿随序列一起 pickle 给其他 rank，作为输入；只有 rank 0 做拒绝采样。

## 正确性验证（CPU）

`tests/engine_sim.py` 的随机模拟器加了两种模式：k=1 和 k=4。假模型现在对一半的请求输出“大致周期性”的文本（草稿常被接受），另一半是伪随机文本（草稿基本被拒）。decode 步的 logits 是 one-hot，交给真实的 `RejectionSampler` 验证，草稿取自输入 tensor，和 `ModelRunner.run` 一致。每个请求的输出必须和逐 token 生成的参考结果完全一致。另外新加了一项检查：每个登记到 prefix cache 的块，KV 必须和它的 token 对得上。

300+30 个用例里的覆盖情况：

| | k=1 | k=4 |
|---|---|---|
| decode 行数 | 20968 | 17391 |
| 接受 0 个草稿 | 72.1% | 81.9% |
| 接受 1 / 2 / 3 个 | 27.9% / - / - | 4.5% / 1.8% / 1.2% |
| 全部接受 | 27.9% | 10.6% |
| 被 max_tokens 截断 | 442 | 592 |
| 被草稿里的 EOS 截断 | 8 | 7 |
| 抢占次数 | 843 | 1259 |

拒绝采样器另有单测：贪心行的结果等于 argmax 链；随机采样时，10 万次采样里每个位置输出 token 的频率（以前面的 token 为条件）和目标 softmax 的差都在 0.02 以内。ngram 有 5 个单测。

我故意往代码里注入了 13 种 bug，全部被抓到（模拟器 330 个投机解码用例里失败的个数，以及单测）：

| 注入的 bug | 模拟器失败 | 单测 |
|---|---|---|
| 拒绝后不回退 `num_cached_tokens` | 165 | 通过 |
| 被拒草稿所在的块也做哈希 | 279 | 通过 |
| 草稿里碰到 EOS 不截断 | 241 | 通过 |
| `can_append` 不算草稿的位置 | 155 | 通过 |
| `may_append` 一步最多加一个块 | 97 | 通过 |
| 草稿的 slot 都算在最后一个块里 | 303 | 通过 |
| `context_lens` 不含草稿 | 316 | 通过 |
| 草稿没传给其他 TP rank | 316 | 5 个失败 |
| 拒绝之后还继续接受（sum 代替 cumprod） | 131 | 3 个失败 |
| 重采样时不去掉被拒的草稿 | 0 | 2 个失败 |
| 接受概率不考虑 temperature | 0 | 1 个失败 |
| bonus token 取自第一行 | 261 | 3 个失败 |
| ngram 匹配到结尾自己 | 316 | 5 个失败 |

后两种采样分布上的 bug 在 one-hot 的模拟器里看不出来，靠分布单测抓。

## 草稿模型（2026-10-23 加）

`LLM(path, num_speculative_tokens=k, speculative_model=draft_path)`：用同一家族的小模型（Qwen3-0.6B 给 Qwen3-8B）打草稿，代替 ngram。验证、拒绝采样、块记账全部复用上面的路径，草稿来源换成 GPU 上的小模型。提交 `962b67d`。

### 做法

```
prepare_decode():  目标模型的输入和之前一样：[t, 0, 0, 0, 0]（草稿位先留空），位置 L-1..L+k-1
                   另外准备草稿模型的输入：[t_{L-2}, t_{L-1}]、位置 L-2..L+k-2 的 slot
propose():         草稿模型跑 k 步：第 1 步输入最后两个 token（每条序列 2 个 query），之后每步输入
                   上一步的 argmax（1 个 query）。每步 argmax 直接写进目标模型输入的草稿位，不回 CPU
run_model():       目标模型一次前向，拒绝采样，和 ngram 版完全一样
```

1. **两套 KV cache，同一张块表**。草稿模型有自己的 `kv_cache`，块数和目标模型相同，块 id 共用：序列的 `block_table` 同时索引两套 cache，prefix cache 命中时两边一起命中。分配块数时按两个模型每块的字节数之和算。Qwen3-0.6B 和 8B 的每 token KV 分别是 115 KB 和 147 KB（都是 8 个 KV 头、head_dim 128，层数 28 对 36），所以草稿 cache 几乎让 KV 内存翻倍，这是草稿模型相对 ngram 的主要代价。
2. **prefill 两边同步**。prefill 时草稿模型用和目标模型完全相同的输入、slot 和块表再跑一遍（不算 logits），两套 cache 在同样的位置上有效。
3. **草稿是贪心的，所以采样器不用改**。草稿模型每步取 argmax，提议分布还是 one-hot，`RejectionSampler` 原样精确。vLLM v1 的 EAGLE/draft-model 路径现在也是这么做的（`draft_probs=None`）。代价是温度采样下接受率比"从 q 采样、按 min(1, p/q) 接受"低一些，好处是不用存 `(B, k, V)` 的草稿概率。
4. **草稿 cache 会落后一个 token**。草稿模型每步只给前 k-1 个草稿算了 KV（第 k 个草稿是它最后一步的输出，没当过输入）。k 个草稿全部接受时，新的 `num_cached_tokens` 指向的最后一个位置在草稿 cache 里就是空的。两处处理：
   - 下一步草稿模型的第一个输入是最后两个 token `[t_{L-2}, t_{L-1}]`，位置 L-2 的 KV 重算一次就补上了。没落后的时候，位置 L-2 的 slot 填 -1，`store_kvcache` 跳过，避免往已经哈希、可能被别的序列共享的块里写。
   - 这个空位所在的块晚一步再做哈希（`Scheduler.postprocess` 里 `hash_blocks(seq, start, num_tokens - seq.draft_cache_lags)`），否则别的序列 prefix cache 命中这个块时，草稿模型会读到一个没写过的位置。`BlockManager.hash_blocks` 加了一个回扫：从 `start // block_size` 往前找到第一个没哈希的块再开始，这样晚一步的块不会被漏掉。`seq.draft_cache_lags` 随序列 pickle 给其他 TP rank，decode 时 pickle 的是最后两个 token 而不是一个。
5. **CUDA graph**。草稿模型抓两套图：每条序列 2 个 query（第一步）和 1 个 query（之后），和目标模型的 k+1 query 图共用输入缓冲区和内存池，按批大小分别抓。`ModelRunner.forward(model, graphs, outputs, ...)` 把"查图、填缓冲、replay"抽成一个方法给两个模型用。
6. **TP**。每个 rank 都跑草稿模型（权重按 TP 切），只有 rank 0 有完整 logits，每步 argmax 后 `dist.broadcast` 给其他 rank（k 次很小的广播）。还没在多卡上跑过。

### 正确性验证（CPU）

模拟器加了 `draft1`、`draft4` 两种模式。假草稿模型是一个可调用对象，塞给真正的 `ModelRunner.propose` 当草稿模型用：它像注意力层一样读全局 context（slot、context_lens、块表），维护自己的假 KV cache（哈希种子和目标模型不同），每次调用检查输入 token、位置、context_len，按 slot 写 KV，再把 context 内每个位置读一遍核对，输出的"隐状态"就是它猜的下一个 token。它猜的 token 有 3/4 的概率等于目标模型的下一个 token。另外加了一条检查：每条序列所有 KV 有效的整块都必须已经哈希（除了草稿模型还没补上的那一块），防止哈希被漏掉。

660 个 draft 用例全过（2005 个用例总计）。k=4 时 9147 个 decode 步里 33% 全部接受，补位置 L-2 的情况出现 2207 次，哈希回扫 90 次；k=1 时分别是 75%、10066 次、604 次。

注入 13 种 bug：

| 注入的 bug | 60 个种子里失败的用例数（共 132） |
|---|---|
| 哈希不延后 | 58 |
| 从不补位置 L-2 | 88 |
| 总是重写位置 L-2 | 102 |
| 草稿步的 context_len 多 1 | 121 |
| 取每条序列的第一行而不是最后一行算 logits | 121 |
| 草稿步用错列（位置和 slot 错位） | 58 |
| `hash_blocks` 不回扫 | 44 |
| `draft_cache_lags` 不 pickle 给其他 rank | 88 |
| 抢占后不清 `draft_cache_lags` | 0 |
| 草稿位置不减 1 | 121 |
| `propose` 之后不恢复目标模型的 context | 121 |
| `draft_cache_lags` 条件取反 | 88 |
| prefill 时不跑草稿模型 | 125 |

唯一没抓到的那种只是多算一次：抢占后重新 prefill，标记还在，第一步多重写一次 L-2 的 KV（值一样），哈希也跟着晚一步，输出和 cache 内容都对，所以模拟器认为没问题。

### 预期

草稿模型一步的开销 ≈ k 次 0.6B 前向。A40 上 0.6B 一步约 3–4 ms（读 1.2 GB 权重），8B 一步约 25–30 ms。k=4 时每步多花 ~15 ms，平均每步多出 2–3 个 token 的话，每 token 时延能降到原来的一半左右；batch 大了以后目标模型本身变成算力受限，多算的 k 个位置不再免费，收益会缩小甚至变负。`cluster/nanovllm-draft-a40.sbatch` 在 Qwen3-8B 上跑 edit/chat × batch 1/8/32 × {无草稿, ngram k=4, 草稿模型 k=2/4}，再加 T=0.6 和 k=6 几组。

## 异步调度 + 草稿模型（2026-10-24 加）

`LLM(path, async_scheduling=True, num_speculative_tokens=k, speculative_model=draft_path)`：三者可以同时开。ngram 版和异步调度仍然互斥（ngram 要在 CPU 上查上一步采样出来的 token，而异步调度的意义就是不等它回来）。

### 难点

异步调度下，调度第 N+1 步时第 N 步还在 GPU 上，CPU 不知道它接受了几个草稿，也就不知道序列现在有多长：下一步的位置、slot、草稿模型的两个输入 token 全都取决于一个还没回到 CPU 的数。

### 做法

1. **在途的是"最多可能加的 token 数"，不是确定的数**。`Scheduler.commit` 给 decode 步记 `num_inflight += 1 + k`（prefill 最后一块记 1），`postprocess` 时再减掉。这个数只用于两处判断：块表够不够（`BlockManager` 按 `len(seq) + num_inflight + k` 算块数）、要不要停（`num_completion_tokens + num_inflight >= max_tokens` 就不再调度，等在途那步落地，避免多跑一步把 token 扔掉）。
2. **decode 步的 KV 在 postprocess 里记账，prefill 块在 commit 里记账**。原来的异步版在 commit 时就把 `num_cached_tokens` 往前推；投机解码接受几个草稿只有落地才知道，所以 decode 步改为在 `postprocess` 里按真实接受数推进（`start = num_cached_tokens`，`num_tokens = seq.num_tokens - start - 1`），prefill 块仍在 commit 时推进（下一块要紧接着排）。`hash_blocks` 用 postprocess 时的 start，commit 时记的 start 对 decode 已经过期。
3. **所有位置都在 GPU 上算**。`prepare_decode_with_draft_model` 把 CPU 知道的序列长度、最后两个 token、`draft_cache_lags` 传上 GPU，然后从上一步输出 `last_token_ids`（形状 `(B, k+1)`，接受的 token 后面补 -1）里数出在途序列实际加了几个 token（`(output >= 0).sum`），据此算出目标模型的 k+1 个位置和 slot、草稿模型的两个输入（最新的 token 和倒数第二个：在途 ≥2 个时都取自输出，恰好 1 个时倒数第二个是 CPU 的 last_token，0 个时和同步版一样）、以及草稿 cache 是否落后（在途序列看 `num_inflight == k+1`，不能用 CPU 上过期的标记）。`run_async` 保证每一步的输出都是 `(B, k+1)`：prefill 步的一维采样结果补 -1 到同样的形状，这样 decode 步索引上一步输出时不用分情况。
4. **`AsyncOutput.result` 做 -1 的剥离**，交给 `postprocess` 的和同步版 `run` 返回的形状一样（每条序列一个 list）。

### 正确性验证（CPU）

模拟器加 `async_draft1`、`async_draft4` 两种模式，假草稿模型和假目标模型照旧核对每次 KV 读写、输入 token、位置。在途检查：模拟器记录每一步的输出，decode 时核对 `num_inflight` 等于 1 或 k+1、在途行的最新 token 就是上一步那条序列的最后一个输出。"KV 有效的整块都已哈希"这条检查原来只在同步模式开，现在异步也开了：在途 prefill 块的 KV 已计入 `num_cached_tokens` 但落地才哈希，检查时减掉 `engine.pending` 里记的那一块。

全部 CPU 用例：2680 个用例（8 种模式 × 300 个随机用例 + 块大小 256 的 30 个 + 单元测试）全过，GPU 用例跳过。

注入 12 种只影响异步路径的 bug（40 个种子，`async_draft1` + `async_draft4` 共 88 个用例）：

| 注入的 bug | 40 个种子里失败的用例数（共 88） |
|---|---|
| commit 时就把 decode 步的 KV 记进 `num_cached_tokens`（结果不错，但 decode 产生的块全部漏掉哈希） | 71（异步模式开"整块必须哈希"检查之前是 0） |
| 停止条件 `>=` 写成 `==` | 66 |
| 没有停止条件 | 77 |
| 在途只记 1 个 token 而不是 1+k | 71 |
| postprocess 用 commit 时记的 start | 0（等价：`num_cached_tokens` 和哈希的块都一样，只是多算了一遍已哈希的块） |
| 已结束的序列仍把在途输出追加进去 | 4 |
| 在途恰好 1 个 token 时倒数第二个 token 取错 | 67 |
| 在途 ≥2 个时倒数第二个取成最新的 | 56 |
| 在途序列的 `draft_cache_lags` 用 CPU 上过期的标记 | 55 |
| 序列长度不加在途数 | 76 |
| 最新 token 不从上一步输出取 | 71 |
| `inflight_row` 记错行 | 26 |

### 预期

异步调度省掉的是每步 CPU 调度和 host→device 拷贝的时间（0.6B 上 10% 左右，8B 上目标模型一步 25–30 ms，占比更小）；草稿模型 k 步小前向本来就在 GPU 队列里，所以两者的收益大致相加。A40 作业 `cluster/nanovllm-draft-a40.sbatch` 里 `async_draft` 模式有 GPU 测试，吞吐对比待加。

## GPU 实测：A40 冒烟（2026-10-24）

Slurm 作业 223221，1 张 A40，Qwen3-0.6B，13.5 分钟。整套测试在 GPU 上跑：2680 个用例全过（模拟器这次用的是真 CUDA 张量），`tests/test_model_gpu.py` 五种配置（同步、异步、ngram、草稿模型、异步+草稿模型）的贪心输出和 transformers fp32 的 teacher-forcing 对比都在 0.5 个 logit 以内，默认容差不用调。

| 负载 | 配置 | 吞吐 | 接受率 |
|---|---|---|---|
| `bench.py` 随机 token，64 条，输入/输出各 ≤256 | 基线 | 6200 tok/s | |
| | 异步调度 | 6842 tok/s（+10.4%） | |
| | ngram k=4 | 1804 tok/s | ≈0 |
| `bench_spec.py --workload edit --batch-size 8` 贪心 | 基线 | 1196 tok/s | |
| | 异步调度 | 1300 tok/s（+8.7%） | |
| | ngram k=4 | 2376 tok/s（×1.99） | 0.796 |
| | 草稿模型（0.6B 给自己打草稿）k=4 | 1023 tok/s | 1.000 |
| | 草稿模型 + 异步调度 k=4 | 1066 tok/s | 1.000 |

怎么读：

- 异步调度在两种负载上都有 9–10% 的提升，和设计说明里"省掉的是每步 CPU 调度和 host→device 拷贝"的估计一致；0.6B 一步只有几毫秒，CPU 开销占比最大，所以这是上限附近的数字，8B 上会小一些。
- ngram 在代码编辑负载上接受率 79.6%，吞吐翻倍；在随机 token 上慢 3.4 倍，因为每条序列都算 k+1 个位置但一个都猜不中。这正是"要老实展示的最坏情况"。
- 草稿模型用 0.6B 给自己打草稿，接受率 1.000 说明草稿 cache、落后一步的哈希、拒绝采样这一整条路在 GPU 上是对的，但两个模型一样大自然不会更快（慢 15%，就是 k 次草稿前向的开销）。真正的加速看下一节 Qwen3-8B 的实测。

## GPU 实测：Qwen3-8B + Qwen3-0.6B 草稿（2026-10-25）

Slurm 作业 225010，1 张 A40，bf16，1 小时 39 分（约 300 SU）。目标模型 Qwen3-8B，草稿模型 Qwen3-0.6B。GPU 测试 18 个全过（含草稿模型和自适应两种模式）。吞吐算整个 run（prefill + decode），单位 tok/s；接受率 = 接受的草稿 / 提出的草稿；倍数相对同一行的基线。"自适应"是草稿 k=4 加 `speculative_disable_by_batch_size=8`。原始 58 行结果在仓库的 `docs/benchmarks/qwen3-8b-a40.txt`。

| 负载 | batch | 基线 | ngram k=4 | 草稿 k=2 | 草稿 k=4 | 草稿 k=4 + 异步 | 自适应 |
|---|---|---|---|---|---|---|---|
| edit 贪心 | 1 | 33.4 | 94.3（×2.8，接受率 0.50） | 72.9（×2.2，0.93） | 96.4（×2.9，0.88） | 97.9 | 96.2 |
| | 8 | 194.7 | 404.4（×2.1，0.52） | 280.5（×1.4，0.91） | 404.3（×2.1，0.86） | 409.1 | 391.2 |
| | 32 | 592.7 | 923.1（×1.56，0.51） | 741.6（×1.25，0.91） | 859.0（×1.45，0.85） | 836.8 | 450.0（×0.76，修前） |
| chat 贪心 | 1 | 33.8 | 40.0（×1.18，0.05） | 56.1（×1.66，0.55） | 59.4（×1.76，0.41） | 60.6 | 60.1 |
| | 8 | 250.1 | 237.2（×0.95，0.05） | 328.4（×1.31，0.57） | 346.5（×1.39，0.41） | 351.5 | 329.4 |
| | 32 | 878.4 | 674.6（×0.77，0.05） | 1019.8（×1.16，0.55） | 965.3（×1.10，0.40） | 984.4 | 746.0（×0.85，修前） |
| edit T=0.6 | 8 | 195.4 | 372.6（×1.91，0.50） | 328.1（×1.68，0.91） | 404.6（×2.07，0.87） | 404.5 | 393.5 |
| chat T=0.6 | 1 | 33.8 | 39.9（×1.18，0.05） | 55.9（×1.65，0.55） | 60.1（×1.78，0.42） | 60.7 | 60.6 |

另外 batch 1 贪心下草稿 k=6：edit 109.0 tok/s（×3.3，接受率 0.82），chat 56.5（接受率 0.31）。单独开异步调度的行都在基线 ±2% 以内（edit 33.4/196.7/583.1，chat 34.2/252.9/888.4）。

怎么读：

- 0.6B 给 8B 打草稿在所有负载、所有 batch 上都更快：batch 1 时代码编辑 ×2.9、聊天 ×1.8；batch 32 还有 ×1.45 和 ×1.16。T=0.6 和贪心几乎一样，拒绝采样没有额外代价。
- ngram 是要老实展示的最坏情况：代码编辑（输出复述输入）×2.8，聊天接受率只有 0.05，batch 8 和 32 反而变慢。
- 异步调度在 0.6B 上 +10%，在 8B 上测不出来：它省掉的是每步固定的几毫秒 CPU 工作，相对 0.6B 的一步很大，相对 8B 的一步很小。这组对比本身就说明了它什么时候有用。
- k 的选择看接受率：edit 接受率 0.85–0.93，k 越大越好（k=6 最快）；chat 接受率 0.4–0.55，batch 32 时 k=2（1019.8）比 k=4（965.3）快，因为没猜中的位置也要算。
- **自适应投机的问题**：阈值以下和草稿 k=4 一样（符合设计），但 batch 32 关掉草稿后比基线还慢：edit ×0.76，chat ×0.85。原因是为了让草稿 cache 不掉队，不带草稿的步仍然跑一次草稿模型（2 个 query），而这一步要读草稿模型的整个 KV cache：Qwen3-0.6B 每层的 KV 和 8B 一样大（8 个 KV 头 × 128），28 层对 36 层，所以这一步的代价远不止"参数量 7%"。edit 的 prompt 更长，慢得更多（每步多 17 ms 对 6.5 ms），也和"读 KV"的解释一致。修法和 GPU 复测见下一节的"自适应 k 的修正"。

## 自适应 k（2026-10-24 加）

`LLM(..., speculative_disable_by_batch_size=n)`：调度 decode 步时，如果正在运行的序列超过 n 条，这一步不带草稿（每条序列 1 个 query），否则照常带 k 个。vLLM 的 `disable_by_batch_size` 是同一个思路。batch 大了以后目标模型本身已经算力受限，多算 k 个位置不再免费，接受率不够高就会变慢（设计说明前面"预期"一节说的情况）。

### 做法

- **调度器按步决定**：`k_step = 0 或 k`，`num_scheduled_tokens = 1 + k_step`，块分配、在途 token 预留（`num_inflight`）、ngram 查找都按 `k_step` 来。两种步交替出现，但同一步里所有序列一样，`(B, q)` 的形状仍然统一。
- **model runner 按步读 q**：decode 时从 `seqs[0].num_scheduled_tokens` 读出这一步的 k；CUDA graph 对 q=1 和 q=1+k 各抓一套（只在设了这个选项时抓 q=1 那套）；采样器按 q 选普通采样或拒绝采样；异步调度下上一步的输出仍然补成 `(B, 1+k)`，所以下一步不用知道上一步带没带草稿。
- **草稿模型的 cache 不能掉队（第一版）**：不带草稿的步如果完全跳过草稿模型，它的 KV cache 就会缺这一步的 token，下次恢复投机时每条序列要补的数量不同（变长，CUDA graph 和 2-query 的第一步都不适用）。所以第一版让不带草稿的步仍然跑草稿模型的第一步（2 个 query：最后两个 token，写入最新 token 的 KV），只是不取它的输出。当时估的代价是"一次小模型前向，0.6B 对 8B 约 7%"；8B 实测（上一节）这一步要读草稿模型整个 KV cache，batch 32 时比不投机慢 15–24%，所以要改成下面的做法。

### 正确性验证（CPU）

模拟器加 `spec4_adaptive`、`draft4_adaptive`、`async_draft4_adaptive`：阈值随机取 1–3，这样带草稿和不带草稿的步大约各占一半（40 个种子里 1:1 左右，两种步在 batch 1–6 都出现过）。假模型按每步的 `num_scheduled_tokens` 核对输入形状、位置、slot，不带草稿的步核对草稿模型只跑了一步、且输入列没被写。

全部 CPU 用例（11 种模式 × 100 个随机种子 + 块大小 256 的用例 + 单元测试，1235 个）全过。

注入 6 种 bug（40 个种子，3 种自适应模式共 132 个用例）：

| 注入的 bug | 失败的用例数（共 132） |
|---|---|
| 不带草稿的步完全跳过草稿模型（cache 掉队） | 51 |
| 不带草稿的步仍把草稿模型的输出写进输入 | 51 |
| 关掉投机时 ngram 的草稿没清掉 | 25 |
| 草稿模型的输入张量按 k 而不是 max(k, 1) 分配 | 51 |
| 不带草稿的步之后不清 `draft_cache_lags` | 0（等价：多重写一次同样的 KV，哈希晚一步） |
| 块分配按配置的 k 而不是这一步的 k | 25 |

### 自适应 k 的修正：草稿 cache 可以掉队（2026-10-25）

不带草稿的步完全跳过草稿模型，让它的 cache 掉队；每条序列记下草稿 cache 覆盖到哪（`seq.num_draft_cached_tokens`，取代原来的布尔 `draft_cache_lags`）。恢复投机的那一步，草稿模型先补齐再打草稿：

- **补齐步的形状仍然统一**：每条序列缺的 token 数不同，但不走变长注意力，而是所有序列都算最后 m 个 token（m = batch 里最大的缺口，至少 2），用和目标模型 `(B, k+1)` 一样的 `flash_attn_with_kvcache` 右对齐因果掩码。已经在 cache 里的位置 slot 置 -1（算了但不写），比 m 短的序列前面补 0（位置夹到 0，输出丢掉）。最后一列的 logits 就是第一个草稿 token，之后 k-1 步照常走 1-query 的 CUDA graph。
- **m=2 时走原来的 2-query 图**：正常投机的步缺 1 或 2 个 token，和改之前完全一样；只有刚从"关草稿"切回来的那一步 m>2，走 eager 前向，这种切换只在 batch 大小跨过阈值时发生。
- **异步调度下在途步的处理**：在途的步如果带草稿，它自己已经补齐过，落地后缺口只会是 1 或 2（全接受时 2），所以 m 的上界按 2 算；如果在途的步不带草稿，缺口 = CPU 已知的缺口 + 在途 token 数。补齐输入在 GPU 上拼：CPU 已知的尾巴 + 在途输出，再按每条序列的在途 token 数 gather 出最后 m 个。TP 的其他 rank 通过 pickle 拿到的是 `token_ids[num_draft_cached:]`，所以尾巴里已缓存的位置统一填 0，各 rank 算出的输入一致。
- **哈希只到两个 cache 都有的地方**：`hash_blocks(seq, start, min(num_cached, num_draft_cached) - start)`；关草稿期间填满的块先不哈希，补齐那一步落地后 `hash_blocks` 的回扫把它们一起补上。前缀缓存命中的块两个 cache 都有，所以分配时 `num_draft_cached = num_cached`；prefill 两个模型一起算，commit 时同步推进。

模拟器相应改成：不带草稿的步不许准备草稿输入；补齐步核对每条序列的位置是 `max(0, n-m..n-1)`、未屏蔽的 slot 和 token 都对、最新 token 一定写入、补 0 的行不写；缺口有没有补全靠假草稿模型读 cache 时逐位置核对。

全部 CPU 用例（1235 个）全过。注入 12 种 bug（6 种草稿模型模式 × 40 个种子 + 块 256 用例，共 264 个用例）：

| 注入的 bug | 失败的用例数（共 264） |
|---|---|
| 在途带草稿的步落地后，全接受时少算一个缺口 | 67 |
| 补齐窗口 m 少算 1 | 20 |
| 已缓存位置的判断 `>=` 写成 `>`（少写一个位置） | 238 |
| 不屏蔽已缓存位置，全部重写 | 229 |
| 后续草稿步的 slot 错位一列 | 154 |
| 后续草稿步的位置错位一列 | 154 |
| 补齐输入里已缓存的位置填真实 token 而不是 0（TP 各 rank 输入不一致） | 3 |
| 不带草稿的步把草稿 cache 标成"已跟上" | 40 |
| 全接受后不记"少一个" | 92 |
| 哈希只看目标模型的 cache | 83 |
| prefill commit 不推进草稿计数 | 217 |
| 抢占时不清草稿计数 | 0（等价：重新 prefill 分配块时会重设） |

**GPU 复测（作业 225388，2026-10-25，1 张 A40，18 分钟）**：18 个 GPU 测试全过（含草稿模型和自适应模式）。Qwen3-8B + Qwen3-0.6B，batch 32，单位 tok/s，自适应 = 草稿 k=4 且 `disable_by=8`，括号里是对同一次运行基线的倍数：

| 工作负载 | 不投机 | 草稿 k=4 | 自适应（修前，作业 225010） | 自适应（修后） | 自适应 + 异步 |
|---|---|---|---|---|---|
| edit | 591.2 | 859.7（×1.45，接受率 0.85） | 450.0（×0.76） | 578.2（×0.98） | 568.8（×0.96） |
| chat | 877.4 | 963.2（×1.10，接受率 0.40） | 746.0（×0.85） | 874.5（×1.00） | 889.1（×1.01） |

batch 8 的 edit（阈值以下，草稿还开着）：草稿 k=4 399.4，自适应 390.0，和修前一样接近。原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-recheck.txt`。

读法：

- 关草稿的步不再跑草稿模型，修前 15–24% 的损失没了。edit 上还剩 2% 左右：每步的调度判断、切回投机时那一次 eager 补齐，或者就是测量波动（两次运行的基线差 0.3%）。
- 更要紧的是阈值本身：这台 A40 上 8B 在 batch 32 仍是访存受限，草稿 k=4 一直开着才是最快的（edit ×1.45，chat ×1.10），"超过 8 条就关"等于白丢加速。阈值应该定在 target 变成算力受限的地方（这里至少 64），更好的开关是实测接受率：chat 上 n-gram 只有 0.05，该关；模型草稿 0.40–0.85，该开。

## 按接受率调 k（2026-10-25 加开关，2026-10-26 改成按草稿位调 k）

`LLM(..., num_speculative_tokens=K, speculative_min_acceptance=0.2)`：每个草稿位单独统计接受率，每个 decode 步只提出前面接受率够阈值的那几位，可以是 0 位（不打草稿）也可以是全部 K 位。n-gram 和草稿模型都能用，可以和 `speculative_disable_by_batch_size` 同时开（batch 太大就直接 0）。起因是两件事：按 batch 大小关草稿的阈值和硬件、模型绑在一起（这张 A40 上 8B + 0.6B 用 8 太低），真正该关的是接受率低的情况（chat 上 n-gram 只有 0.05，batch 8 以上反而变慢）；而 8B 的实测里最优 k 随负载变（edit 上 k=6 最好，chat 上 batch 32 时 k=2 比 k=4 好），一个开关不够，要调 k。第一版（14df55e）只是开关：全 batch 的接受率低于阈值就 k=0；这一版把它推广成按位调 k，开关是它的特例。

### 做法

- **统计**：第 j 位被接受的前提是前 j-1 位都被接受，所以第 j 位的接受率天然随 j 递减，"砍到哪一位"就是看它从哪一位开始低于阈值。`Scheduler.postprocess` 对每条投机序列、它这一步提出的每个位 j 记一个样本（提出了；是否接受：`j < len(new_token_ids)`），每位维护最近约 `ACCEPTANCE_WINDOW`（64）个序列-步的平均，但窗口至少覆盖 `ACCEPTANCE_MIN_STEPS`（8）步（batch 32 时 64 个样本两步就换完，噪声 ±0.05，k 会在阈值附近来回跳；2026-10-26 加）：没看满窗口前是普通平均，之后每步的样本按份额替换（`acc[j] = (acc[j]·seen + accepted[j]) / (seen + proposed[j])`，`seen` 封顶在窗口大小）。只提出了 k<K 位的步不改后面几位的统计。`reset_stats` 另外累计每个位提出/接受的总数和每步 k 的分布，`bench_spec.py` 打印成 `acc_pos=` 和 `k_steps=`。
- **决定**：`schedule` 取 k = 最长的前缀使 `acc[1..k]` 都不低于阈值（中间断一位就停，后面的位不单独算）。n-gram 提案截到前 k 个，草稿模型只跑 k 步。每 `PROBE_INTERVAL`（16）个 k<K 的 decode 步插一个 K 位的探测步，重新测量被砍掉的位，探测步里后面几位接受率高就回来了。没测过之前信任（初始 1.0）。batch 大小规则说 0 就 0，不再看接受率；探测计数照常累加，batch 变小后的第一步就是探测步。空调度步不推进探测计数。
- **模型侧**：以前 decode 步只有 q=1 和 q=K+1 两种形状，现在 1..K+1 都可能：开了 `speculative_min_acceptance` 时 target 给每个 q 各捕一套 CUDA graph，草稿模型的追赶/生成 graph 不变（q∈{2,1}）。异步调度下在途步带的草稿数不再等于 K：`prepare_decode_with_draft_model` 用 `seq.num_inflight > 1` 判断在途步带草稿、用"产出 token 数 == seq.num_inflight"判断全接受（原来写死 `1+K`），`postprocess` 里"全接受导致草稿 cache 少一个 token"的判断也改成按这一步实际的草稿数。
- **采样器编译**：`Sampler` 和 `RejectionSampler` 用 `torch.compile`，原来按具体形状编译，每个新的 (batch, k) 形状都在计时区间里重新编译几秒；自适应 k 让一次跑里出现 q=1..K+1 所有形状（加上尾批的各种 batch），编译开销能吃掉百分之十几的吞吐（228429 作业里 chat 上 n-gram 开了开关反而比不开慢 5%，怀疑就是这个）。改成 `dynamic=True` 只编译一次，并在启动时对每个 q 预热（`ModelRunner.warmup_samplers`），和 CUDA graph 的 q 集合共用 `decode_query_lengths`。
- 顺带修了 14df55e 的一个 GPU bug：只开 `speculative_min_acceptance`、不开 `speculative_disable_by_batch_size` 时没有捕 q=1 的 graph，第一个关草稿的步会 KeyError。CPU 测试不走 CUDA graph，没抓到；现在 graph 的 q 集合按两个开关算。
- **代价**：k<K 期间每 16 步一次探测。n-gram 在 chat 上的探测步每条序列多算 K 个位置（batch 32 时那一步慢约 30%），平摊到 16 步约 2%；草稿模型的探测步还要先补齐草稿 cache（最多 17 个 token 的 eager 前向）。

### 正确性验证（CPU）

- `tests/test_acceptance_switch.py` 直接驱动调度器（12 个用例）：没测过先信任；从第一个被拒的位往后砍；中间断一位就停；每隔 PROBE_INTERVAL 用全 K 位探测一次；短步计入探测计数、不改后面位的统计；探测成功后位回来；窗口会忘掉旧历史；按位、按整个 batch 统计；和 batch 大小规则叠加；阈值 0 永不砍；窗口至少 8 步；统计按步和按位计数。
- 模拟器的 `spec4_acc`、`draft4_acc`、`async_draft4_acc` 三种模式：阈值在 {0.2, 0.5, 0.8} 里随机，窗口 2–40、探测间隔 1–6 随机；假草稿模型按位分质量：一半请求全不命中，另一半第 1 位 7/8 命中、第 2 位 3/4、之后 1/2，所以 batch 里各位的接受率随成员和位置变，k 在 0..4 每个值都停过（60 个种子里每种模式约 1100 个 decode 步，草稿模型模式 k=0 约 540、k=1 约 90、k=2 约 90、k=3 约 33、k=4 约 340）。策略检查改成：模拟器自己从调度器的接受率数组算出这一步应有的 k（独立实现：最长不低于阈值的前缀；探测计数到了就是 K），每个 decode 步必须相等；在途 token 数允许 1..1+K。追赶步的检查改成精确的：m 必须正好是 2 或草稿 cache 缺得最多的那条序列缺的数，slot 必须恰好在已缓存的位置屏蔽（在途带草稿的步算它补过之后的 cache），多写一个 token 也算错（上一版只查"缺的都写了"，抓不住多算的）。KV 读写、哈希块、输出和参考完全一致的检查照旧。
- 全部 CPU 用例（14 种模式 × 100 个种子 + 单元测试，共 1577 个）全过。

注入 21 种 bug，跑 12 个单元测试 + 132 个模拟器用例（3 种 `_acc` 模式 × 40 个种子 + 块 256 用例）：

| 注入的 bug | 失败的用例数（共 144） |
|---|---|
| 从不探测（砍掉的位永远不回来） | 101 |
| 探测计数器从不清零 | 108 |
| 只提出了部分位的步也算探测（计数器清零） | 52 |
| 探测间隔差一（`<=`） | 101 |
| 初始接受率 0（没测过就不信任草稿） | 11 |
| 阈值比较用 `>` 而不是 `>=` | 42 |
| 中间断一位不停（数所有够阈值的位） | 1 |
| 窗口不保证至少 8 步（大 batch 两步就换完） | 1 |
| 每步 k 的统计只数探测步 | 1 |
| 窗口不封顶（变成累计平均，忘不掉历史） | 2 |
| 窗口只看当前步（没有历史） | 7 |
| 接受数把采样出的那个 token 也算上 | 9 |
| 没提出的位也算成被拒 | 4 |
| 空调度步也推进探测计数器 | 23（只有异步模式抓到：探测被空步吃掉） |
| batch 大小规则已经说 0 时接受率规则再按位给 k | 1 |
| n-gram 提案不截到 k 个 | 15 |
| 全接受时草稿 cache 从不掉队（`lag` 恒为假） | 17 |
| 只有 K 位全接受才算掉队（k<K 的步全接受算漏） | 15 |
| 追赶长度 m 只对在途 K 位的步取 2（k<K 的在途步按整段算） | 14（精确的追赶检查才抓到） |
| 在途步只有带满 K 位才算"自己补过 cache"（k<K 的在途步按旧 cache 屏蔽 slot） | 13（精确的追赶检查才抓到） |
| "全接受"按 K+1 个 token 判断（k<K 的在途步） | 10 |

### GPU 实测（作业 228429，2026-10-26，1 张 A40，53 分钟，代码 8c4dde3）

预期是四件事：chat 上 n-gram k=4 加 `min_acceptance=0.2` 应回到不投机的水平（batch 8/32 修前是 ×0.95 / ×0.77）；edit 上 n-gram 和 chat 上草稿模型的加速必须保住；混合 batch 看它怎么选；草稿模型 K=6 加 `min_acceptance=0.3` 时 edit 应接近固定 k=6 的 109 tok/s（batch 1），chat 应自动落到 k=2–3。实测（24 个 GPU 测试全过；原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-k.txt`；基线列来自作业 225010，同一配置两次运行能差 10%：chat batch 32 的 n-gram k=4 那次 674.6、这次 763.8）：

| 负载 | batch | 基线（225010） | 草稿 k=6 固定 | 草稿 K=6 + min_acc 0.3 | 草稿 K=4 + min_acc 0.2 |
|---|---|---|---|---|---|
| edit | 1 | 33.4 | 109.0（×3.3，接受率 0.82） | 123.9（×3.7，0.82） | |
| edit | 32 | 592.7 | 818.7（×1.38，0.79） | 969.5（×1.64，0.79） | |
| chat | 1 | 33.8 | 56.5（×1.67，0.31） | 60.1（×1.78，0.52） | |
| chat | 32 | 878.4 | 868.6（×0.99，0.31） | 1104.8（×1.26，0.52） | 1108.2（×1.26，0.48） |

- **草稿模型上按位调 k 是干净的赢面**：每一格都比固定的 k=4 和 k=6 快。chat batch 32 上比固定 k=6 快 27%、比固定 k=4（965.3）快 15%：后面几位平均只有 31% 被接受，砍掉之后每步少算几个位置，探测步每 16 步一次。edit batch 1 从 109.0 到 123.9，是这台 A40 上 8B 单请求的最好数字（×3.7）。接受率从 0.31 升到 0.52 正是"砍掉了不该提的位"的样子。
- **n-gram + 开关保住了 edit 的加速**（batch 32 1002.8、batch 1 111.5，都高于 225010 的 923.1 / 94.3），**但 chat 和混合负载没回到基线**：chat batch 32 729.0（不开开关 763.8，基线 878.4）、batch 8 267.3（基线 250.1，这个回到了）；混合 batch 32：不投机 842.9、n-gram 719.1、n-gram + 开关 727.2、草稿模型 925.6、草稿模型 + 开关 871.6；混合 batch 8：251.1 / 247.9 / 222.2。
- 事后在代码里找到两个原因，已在 44129e2 修掉：（1）`Sampler` / `RejectionSampler` 的 `torch.compile` 按具体形状编译，自适应 k 让计时区间里出现 q=1..K+1 所有形状，每个新形状重编译几秒（chat batch 32 一行总共才 42 秒）；改成一次动态编译并在启动时预热。（2）窗口 64 个样本在 batch 32 时两步就换完，位 1 的接受率估计噪声 ±0.05，正好在 0.2 阈值附近来回跳，k 在 0 和 1 之间抖；改成窗口至少 8 步。还有一个是设计上的限制：一步只有一个 k，整个 batch 共用，混合负载里 edit 一半接受率高、chat 一半低，平均下来 k 砍不掉，chat 那一半白算——要解决得每条序列一个 k（ragged 的 query，vLLM 的做法），这是下一步。
- 这一节的基线列来自另一次作业，下一节的复测证明这样比是不对的：复测那台 A40 的每个基线都比 225010 高 13–17%，这里的 ×3.7、×1.64、×1.26 都偏大，以下一节同场的数字为准。

### GPU 复测（作业 228654，2026-10-27，1 张 A40，64 分钟，代码 1f8127f）

脚本 `cluster/nanovllm-adaptive-k2-a40.sbatch`：29 行，每种负载/batch 先跑一行不投机的基线，所有对比都在同一次作业里；每行打印各草稿位的接受率 `acc_pos=` 和每步 k 的分布 `k_steps=`。24 个 GPU 测试全过；原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-k2.txt`。少了一行（chat batch 32 n-gram 加 `disable_by_batch_size=0`）：Config 要求这个值 ≥ 1，而脚本把 bench 的 stderr 丢了，所以日志里没有报错——下个脚本改成 `=1` 并保留 stderr。单位 tok/s，括号里是对本次基线的倍数和接受率。

| 负载 | batch | 基线 | 草稿 k=4 固定 | 草稿 K=4 + min_acc 0.2 | 草稿 k=6 固定 | 草稿 K=6 + min_acc 0.3 |
|---|---|---|---|---|---|---|
| edit | 1 | 38.4 | | | 124.3（×3.24，0.82） | 123.7（×3.22，0.82） |
| edit | 32 | 683.6 | 978.2（×1.43，0.85） | 970.8（×1.42，0.86） | 975.4（×1.43，0.79） | 963.9（×1.41，0.79） |
| chat | 1 | 39.0 | | | 63.6（×1.63，0.31） | 63.4（×1.63，0.52） |
| chat | 32 | 1025.3 | 1065.3（×1.04，0.40） | 1222.2（×1.19，0.48） | 854.1（×0.83，0.31） | 1103.4（×1.08，0.50） |
| 混合 | 32 | 847.1 | 798.9（×0.94，0.57） | 909.1（×1.07，0.57） | | |

| 负载 | batch | 基线 | n-gram k=4 固定 | n-gram k=1 固定 | n-gram K=4 + min_acc 0.2 | 开关下每步 k 的分布（k=0..4） |
|---|---|---|---|---|---|---|
| edit | 32 | 683.6 | | | 1051.5（×1.54，0.51） | 2, 0, 0, 0, 405 |
| chat | 1 | 39.0 | | | 39.5（×1.01，0.13） | 3268, 237, 0, 0, 219 |
| chat | 8 | 289.0 | | | 271.9（×0.94，0.10） | 936, 15, 0, 0, 60 |
| chat | 32 | 1025.3 | 754.3（×0.74，0.05） | 922.1（×0.90，0.14） | 902.8（×0.88，0.10） | 928, 23, 0, 0, 60 |
| 混合 | 32 | 847.1 | 715.6（×0.84，0.17） | | 746.8（×0.88，0.27） | 523, 144, 79, 11, 241 |

- **草稿模型上，开关在有位被拒的地方赚、其他地方不亏。** chat batch 32：K=4 从固定的 ×1.04 到 ×1.19（453 步里 380 步停在 k=3），K=6 从亏（×0.83）到 ×1.08（540 步里 331 步 k=2、141 步 k=3）；混合 batch 32 从 ×0.94 到 ×1.07。edit 上各位接受率 0.72–0.92 都过阈值，开关一直给 k=K，和固定 k 差 1% 以内。chat batch 1 上开关把 K=6 砍到多数步 k=2，吞吐纹丝不动（63.4 对 63.6）：batch 1 一步读一遍权重，q 多几个不要钱，砍掉的位本来就是免费的，和设计时的预期一致。
- **上一节 edit batch 1 "109.0 → 123.9（×3.7）" 是错的**：109.0 是 225010 里的固定 k=6，123.9 是 228429 里的自适应，两次作业的基线差 15%。同场测：固定 124.3、自适应 123.7，都是 ×3.2，开关在这格不赚也不亏（本来也不该赚：每一位都过阈值）。
- **n-gram + 开关：edit 保住 ×1.54，chat batch 1 回到基线（×1.01），batch 8（×0.94）和 32（×0.88）还差一截，混合 ×0.88。** 44129e2 的两处修复是有效的：228429 里开开关比不开还慢（729 对 764），这次是 903 对 754。chat batch 32 这一行 1011 步里 928 步 k=0、60 步探测（k=4）：探测步约是普通步的 1.5 倍，只能解释 12 个点里的 3 个；k=0 的步走的是和基线一模一样的代码路径（`prepare_decode`、`Sampler`、q=1 的 CUDA graph），剩下的 9 个点现在解释不了。下一个作业单独量这条路径（n-gram 加 `disable_by_batch_size=1`：每步都不出草稿、也不探测），并在一次作业的末尾再跑一遍基线，看同一作业内前后的漂移有多大。
- **混合负载上开关在抖**：batch 平均的各位接受率 0.31/0.27/0.25/0.24 刚好在 0.2 阈值上方，于是 523 步 k=0、241 步 k=4 来回换，而 k=4 的步里 chat 那一半的草稿照样全算全拒。这就是"一步一个 k"的设计限制，每条序列一个 k（ragged 草稿）是下一阶段。

### 剩下的 12 个点在哪（作业 230846，2026-10-28，1 张 A40，17 分钟，代码 6f10641）

脚本 `cluster/nanovllm-adaptive-k3-a40.sbatch`，只跑 chat，11 行，原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-k3.txt`；每行之前打印 GPU 温度/频率/功耗。

| 行（chat，batch 32） | tok/s | 对基线 |
|---|---|---|
| 不投机（第 1、5、7 行） | 878.9 / 876.4 / 879.2 | — |
| n-gram k=4 + `disable_by_batch_size=1`（每步都不出草稿、不探测，但 proposer 在） | 878.4 | ×1.00 |
| n-gram K=4 + min_acc 0.2，每 16 步探测（60 次探测 + 23 步 k=1） | 827.4 | ×0.94 |
| n-gram K=4 + min_acc 0.2，每 64 步探测（15 + 12） | 833.3 | ×0.95 |
| n-gram k=4 固定 | 679.0 | ×0.77 |

batch 8：基线 250.2 / 250.2，关草稿 250.1，开关 235.8（×0.94）。

- **同一作业内没有漂移**：三次基线差 0.3%，频率一直 1740 MHz、41–44 ℃、76–81 W。两次作业之间却差 15%（这台 878.9，228654 那台 1025.3，代码和参数一样），所以只能同作业内比。
- **关草稿的路径本身不要钱**：878.4 对 878.9。228654 里怀疑的"k=0 的步比基线慢"不成立。
- **开关的损失基本不随探测次数变**：探测减到 1/4，损失只从 5.9% 变到 5.2%。按 k=4 固定那行算，一步 k=4 是 48.3 ms、普通步 34.6 ms，60 次探测只多 0.8 s，而开关那行总共多了 2.4 s；探测 64 那行多 2.0 s，探测只解释 0.2 s。两行都剩约 1.5–1.8 s 的固定开销，和出了多少草稿无关。
- **固定开销是 `torch.compile` 的 0/1 特化**：dynamic=True 把每个维度当符号，唯独大小 0 和 1 会特化成常数。启动时的预热只用 batch 8 编译，bench 自己的预热（1 个 prompt、16 个 token）又把 batch 1 的 q=1 和 q=K+1 编了，但 batch 1、q=2（一条序列、一个草稿）谁都没编——每个 batch 的尾巴都会收缩到一条序列，碰上一步 k=1 就在计时区间里重编译一次（GPU 上约 1.5 s）。CPU 上开着 dynamo 的重编译日志复现了这条序列：全程 6 张图，其中 1 张在尾巴那步才编。第一版修法是 `warmup_samplers` 把 batch 1 也预热一遍（c4cf5ac），CPU 上同一条序列计时区间里零编译。
- **GPU 上复测（作业 231468，开着 `TORCH_LOGS=recompiles`）却没好**：开关行 961.9 对基线 1038.6（×0.93）。日志里 186 条 Recompiling 说明了原因：每一行（包括基线）在第一个真实请求时把所有编译过的层（两个 layernorm、rotary、activation）和两个采样器**全部重编译一遍**，守卫是 `utils_device.CURRENT_DEVICE == device('cuda')`——`ModelRunner` 加载模型和预热时把 torch 的默认设备设成 cuda，服务时又改回 cpu，而 `torch.compile` 会对默认设备加守卫。所以启动时的预热全部作废；bench 自己的预热调用把这一串重编译挡在了计时器外面，但那一次是 batch 1 的 prompt，于是采样器被 0/1 特化成 batch 1，计时区间里再重编译两次（`temperatures` 从 1 到 32；`2 <= draft_token_ids.size()[1]`，k 从 ≥2 到 1）。对真实用户来说这是第一个请求的延迟：十几次编译、几十秒。
- **第二版修法**：预热和 CUDA graph 捕获都挪到默认设备恢复成 cpu 之后，预热用的张量显式写 `device="cuda"` 和模型的 dtype（ac3f3be）。
- **复测（作业 231709，`docs/benchmarks/qwen3-8b-a40-adaptive-k5.txt`）**：默认设备那条守卫没了（日志从 186 条降到 134 条，层不再重编译），但开关行还是 ×0.94（829.7 对基线 880.2 / 880.8，关草稿 881.0）。剩下的守卫是 `requires_grad mismatch`：`ModelRunner.run` / `run_async` 没有放在 `torch.inference_mode` 下（只有模型的 forward 在），所以真实步的 lm_head logits 带着 autograd、预热的不带，两个采样器在第一个真实步又编了一遍；之后照旧被 bench 的 batch 1 预热 0/1 特化，再加一条 stride 守卫（真实草稿是 `input_ids.view(B, q)[:, 1:]`，stride 是 (q, 1)，预热用的是连续张量），拒绝采样器在计时区间里又编了两次。草稿模型那两行照样是 ×1.08（固定 k=4，953.4）和 ×1.20（K=4 + 开关，1052.4）。
- **第三版修法**：`run` / `run_async` 加 `@torch.inference_mode()`，采样器预热也在 inference mode 下跑，草稿用 `(batch, q)` 张量的列切片。CPU 上开着重编译日志：预热编 6 张图，之后 bench 式的服务序列零编译。
- **复测（作业 232099，2026-10-28，代码 0e2b732，`docs/benchmarks/qwen3-8b-a40-adaptive-k6.txt`）**：开关行 871.8 对基线 881.2 / 881.6（×0.99，231709 里是 ×0.94），batch 8 上 249.9 对 251.6（×0.99，原来 ×0.94），关草稿 881.4 照旧免费；草稿模型固定 k=4 1000.3（×1.14）、K=4 + 开关 1060.7（×1.20）。重编译日志 117 条，没有一条 `requires_grad`，采样器的编译全部落在每行启动后 13 秒内（启动预热里 batch 1 和草稿宽 1 的 0/1 特化，以及它们之后的符号图），计时区间里一次都没有——231709 里开关行是在 +74 s 和 +87 s 重编译的。到这里 228654 那 12 个点算清了：7 个点是两台 A40 的差别，4 个点是启动编译漏进计时区间，1 个点是探测。

## 每条序列一个 k（ragged 草稿，2026-10-29）

`LLM(..., num_speculative_tokens=K, speculative_min_acceptance=0.2, speculative_per_sequence=True)`：接受率按序列统计，每条序列自己决定这一步提几位草稿，一个 batch 里各序列的草稿数可以不同。起因是上一节的混合负载：batch 平均的接受率 0.31/0.27/0.25/0.24 刚好在阈值上方，开关在 k=0 和 k=4 之间抖，k=4 的步里 chat 那一半照样全算全拒。每条序列一个 k 是 vLLM 的做法（varlen 注意力）；这里的原则还是不开开关时一切不变。

### 做法

- **统计和决定（调度器）**：每条序列有自己的 `acceptance[1..K]`。窗口 `ACCEPTANCE_WINDOW_PER_SEQUENCE` = 16 步（一条序列每步每位只有一个样本，满窗口后每步按 1/17 替换；窗口小是故意的，一条序列从“抄 prompt”切到“自由生成”要在几步内被发现）。每个 decode 步对每条序列算 `num_acceptable_drafts(seq)`：最长的够阈值的前缀，没测过先信任（新序列第一步提满 K），但信任背后垫着 `ACCEPTANCE_PRIOR_PER_SEQUENCE` = 1 个“接受”的先验样本：一条序列每步每位只有一个样本，没有先验时开头一次没中这一位就是 0，要等下一次探测（16 步）才回来；有先验是 1/2，再没中一次才砍。先验是在查作业 233732 的 GPU 混合 batch 用例（抄写任务加四个问题，必须出现 ragged 步，结果一个都没有）时加的，当时以为是抄写任务开头一次没中就丢了 16 步草稿；先验本身该有，但不是那个用例的原因——作业 234187 带着先验照样 0 个 ragged 步。真正的原因是 GPU 测试的 prompt 开着 Qwen3 的思考模式（`bench_spec.py` 早就 `enable_thinking=False`，测试的 `chat()` 没有）：抄写任务先想满 64 个 token，根本没开始抄，n-gram 自然一个都不中，五条序列全是 0 草稿，形状当然统一。用例改成关思考，作业 234580（只跑 GPU 测试，代码 0d09bd7，11 分钟）验证：44 个通过、8 个跳过，混合 batch 用例在 pad 和 varlen 两种模式下都出现了 ragged 步。探测还是整个 batch 的：`PROBE_INTERVAL` 个“有序列没提满 K”的步之后，所有序列同一步都提 K，这是一个统一形状的步，走 CUDA graph。第一版（5b62af5）让每条序列各自计数、各自探测，A40 作业 233573 里 chat 上 21% 的步因此变成 ragged 步（各序列的探测落在不同的步，每个都是 eager 步），n-gram 开关从 ×0.99 掉到 ×0.95；改成 batch 同步之后探测步和 batch 级开关一样。`num_scheduled_tokens = 1 + num_drafts_i`，块分配按各自的数，n-gram 提案截到各自的数。batch 大小规则优先（超过就全部 0）。batch 平均的接受率照旧统计（`acc_pos=` 还是它），`num_steps_by_k` 在这个模式里数的是序列-步；另加 `num_decode_steps` / `num_ragged_steps`（各序列草稿数不同的步）/ `num_capped_steps`（被 batch 大小规则清零的步），bench 打印成 `ragged=` 和 `capped=` 两个比例。
- **输入布局（`ModelRunner.ragged_layout`）**：decode 步的输入按序列拼成一维，序列 i 占 `1 + num_drafts_i` 行（最后一个 token，然后它的草稿）。布局函数返回三个张量：`token_index (B, 1+max k)`，每条序列各行的行号，不够的重复它最后一行；`write_index (B, max k)`，第 i 个草稿写到哪一行，没有这个草稿的序列写到末尾多出的一行 scratch 行；`num_drafts (B,)`。所有序列草稿数相同的步（含全 0）就是原来的 `(B, k+1)` 网格，走原来的路：`flash_attn_with_kvcache`、CUDA graph、按 view 切草稿的拒绝采样。只有草稿数不同的步（ragged 步）走下面的新路。
- **注意力**：ragged 步设置 `cu_seqlens_q`，注意力层走 chunked prefill 那条路：`flash_attn_varlen_func` 带 `block_table` 读整个 KV cache，causal 掩码右下角对齐，每条序列的 q 长度不同也没关系。`cu_seqlens_q` 在 CPU 算；`cu_seqlens_k` 在 GPU 上从 `context_lens` 累加，因为异步调度下在途 token 数只有 GPU 知道（CPU 给一个上界当 `max_seqlen_k`）。ragged 步不走 CUDA graph（eager），这是它的代价，要在 GPU 上量。
- **拒绝采样器**：多一个分支，输入 `token_index` 和 `num_drafts`：先按 `token_index` gather 出 `(B, k+1, V)` 的 logits（补齐的位重复最后一行），正常算接受，再把补齐的位强制算拒绝（`accepted &= arange(k) < num_drafts`），“全接受要从最后一行采 bonus”按各自的 `num_drafts` 判断。每条序列的输出和它单独放在一个 k=num_drafts_i 的统一 batch 里完全一样（有测试逐条比）。
- **草稿模型**：追赶步只补有草稿的序列，0 草稿的序列这一步整个跳过草稿模型（slot 全屏蔽），cache 掉队，和它处在 k=0 的步一样，`num_draft_cached_tokens` 的记账不用改。生成第 i 个草稿时，草稿数不到 i 的序列算一个空行：输入是它自己的最后一行（位置夹住，不越界读块表），slot 置 -1 不写 KV，context_len 是它自己的；结果按 `write_index` 写到 scratch 行。target 的前向只取前 n 行（不含 scratch 行）。n-gram 路径更简单：提案已经各自截断，只要布局和 context。
- **异步调度**：在途输出 `last_token_ids` 照旧 `(B, K+1)` 填 -1；ragged 步的输出宽度是这一步最大的 k+1，不够 K+1 的补到 K+1。在途步的草稿数由 `seq.num_inflight` 给出，每条序列不同也没问题。

### 正确性验证（CPU）

- `tests/test_per_sequence_k.py` 直接驱动调度器（9 个用例）：每条序列自己的 k；batch 的统计不漏到序列上；新序列从满 K 位开始；探测是 batch 的一个统一步（有序列没提满 K 的步计数，到了所有序列同一步提 K）；短步计入探测、不改后面的位；窗口会忘掉旧历史（满窗口后每步 1/5，窗口 4）；batch 大小规则对每条序列都生效；统计按序列-步和按 ragged 步计数；n-gram 提案按序列截断。
- `tests/test_ragged_sampler.py`（5 个用例）：布局函数本身；贪心下 ragged batch 的每一行和“把这条序列单独放进它自己 k 的统一 batch”逐条相等；全部相同的 batch 两条路结果一样；温度采样下（T=0.7 和 1.0，5 万对序列）带草稿和不带草稿的序列各自的输出频率都在目标分布的 1% 以内，被拒绝之后的位全是 -1。
- 模拟器加 5 种模式：`spec4_perseq`、`draft4_perseq`、`async_draft4_perseq`、`spec2_perseq`、`draft4_adaptive_perseq`（最后一个叠加 batch 大小规则），窗口 1–12、探测间隔随机。假模型对每条序列独立算出它应有的 k（和 `_acc` 模式一样的独立实现，只是按序列），每步核对；ragged 步核对输入的行偏移、布局张量和 `ModelRunner.ragged_layout` 一致、scratch 行存在且不进 target；统一步必须没有 `cu_seqlens_q` 和采样器布局；追赶步要跳过 0 草稿的序列；假草稿模型对空行核对“没有 slot、context_len 是自己的、位置不越界”，并给它一个固定输出。context 里的 `slot_mapping` / `context_lens` / `cu_seqlens_*` 必须是 int32（flash-attn 的要求；作业 233573 抓到的 bug，见下）。KV 读写、哈希块、输出和参考完全一致的检查照旧。GPU 测试加 `spec_per_sequence`、`draft_per_sequence`、`async_draft_per_sequence` 三种模式和一个混合 batch 用例（一条“原样复述这段文字”的抄写任务加四个问题：抄写任务的 n-gram 草稿留着、问题的被砍掉，必须出现 ragged 步，输出逐 token 对参考）。
- 全部 CPU 用例 2142 个全过（19 种模式各 100 个种子加块 256 用例，单元测试 52 个）。

注入 15 种 bug，跑 14 个单元测试 + 220 个模拟器用例（5 种 `_perseq` 模式 × 40 个种子 + 块 256 用例，共 234；在第一版 5b62af5 上跑的，两条探测计数的注入针对的是当时每条序列自己的计数器）：

| 注入的 bug | 失败的用例数（共 234） |
|---|---|
| 每条序列的接受率从不更新（永远用初始值） | 8（只有单元测试抓到：模拟器按调度器自己的接受率数组算预期 k，数组不动它也不动） |
| 每条序列从不探测（计数器永远清零） | 120 |
| 探测计数器从不清零 | 135 |
| 每条序列的窗口不封顶（累计平均） | 2 |
| 阈值比较用 `>` 而不是 `>=` | 31 |
| n-gram 提案不按序列截断 | 69 |
| 调度 token 数按 batch 最大 k 算（块分配多算） | 178 |
| 拒绝采样器不屏蔽补齐的草稿位 | 19 |
| 拒绝采样器"全接受"按最大 k 判断（补齐位全过时不从最后一行采样） | 19 |
| `token_index` 不按序列的草稿数夹住（读到下一条序列的行） | 37 |
| `write_index` 把没有的草稿写进下一条序列的行 | 26 |
| n-gram 的 ragged batch 当成统一形状（不设 varlen context） | 27 |
| 草稿模型追赶时把 0 草稿的序列也补上 | 23 |
| 草稿模型给草稿少的序列的空行也写 KV | 24 |
| target 的前向把 scratch 行也算进去 | 0：模拟器自己拼 decode 步（prepare、propose、采样器），不经过 `ModelRunner.decode_logits`，看不到这一刀；GPU 测试走真实路径（n+1 行对 n 个 slot，写 KV 时就错），留给作业抓 |

### 预期

- 混合负载（edit 和 chat 各半，batch 32）：n-gram 应是 edit 一半 k=4、chat 一半 0；草稿模型应是 edit 一半 k=4、chat 一半 2–3。开关不再抖，`ragged=` 应接近 1。
- 代价是 ragged 步没有 CUDA graph。8B 在 batch 32 下 eager 比 graph 慢多少，之前没单独量过；bench 有 `--enforce-eager`，作业里加一行 eager 的基线就能看出 graph 值多少。如果 eager 的损失吃掉混合负载的收益，下一步是按总行数分桶抓 CUDA graph（输入 pad 到桶大小，布局张量放进缓冲区），varlen kernel 本身可以进 graph。
- 纯 edit 和纯 chat 上应和 batch 级开关一样（各序列的 k 大多相同，ragged 步少），这是回归检查。

### 实测（作业 233573，2026-10-29，代码 5b62af5，49 分钟）

脚本 `cluster/nanovllm-perseq-a40.sbatch`，GPU 测试加三种模式，18 行 bench（混合负载 batch 32：基线、eager 基线、n-gram 开关 batch 级 / 按序列、草稿模型固定 k=4、开关 batch 级 / 按序列；chat、edit batch 32 和混合 batch 8 做回归；最后再跑一次基线看漂移）。原始行在 `docs/benchmarks/qwen3-8b-a40-perseq.txt`。结果有三条教训，两处 bug：

- **草稿模型的 ragged 步在 GPU 上崩了**：`seqlens_k must have dtype int32`。`propose` 里每一步的 context_len 是 `context_lens - num_drafts + min(num_drafts, i)`，`num_drafts` 是 int64，减出来就是 int64，而 `flash_attn_with_kvcache` 要 int32。统一形状的路是 `context_lens - k + i`（Python 整数），dtype 不变，所以以前没碰到。CPU 上假模型只看值不看 dtype，没抓到；现在模拟器核对 context 里所有 int32 张量的 dtype（去掉修法后 4/12 个用例立刻报错）。GPU 测试里 `draft_per_sequence` 的贪心用例过了（草稿模型给自己打草稿，阈值 0.9 之下没出现 ragged 步），采样用例崩了，之后 `async_draft_per_sequence` 三个用例因为崩掉的引擎没释放显存在建引擎时就失败；两行草稿模型按序列的 bench 也没出数。
- **按序列探测把 chat 上 21% 的步变成了 ragged 步**：chat batch 32 上 n-gram 开关按序列是 836.5 对基线 880.5（×0.95，batch 级在 232099 里是 ×0.99）。各序列的探测计数在第一次不一致之后就错开，每条序列在自己的步上提满 K，和其他序列（k=0）凑成 ragged 步，走 eager。改成 batch 同步探测（见上面的做法）。
- **eager 的代价在 batch 8 上压过了收益**：混合负载 batch 8，草稿模型开关 batch 级 310.2（×1.43），按序列 275.3（×1.27），80% 的步是 ragged 步。batch 32 上 eager 基线比 CUDA graph 基线慢 4%（692.6 对 720.2），batch 8 上没单独量（第二次作业加了这一行），但 8B 在 batch 8 一步只有二三十毫秒，几十个 kernel 的启动开销占比更大。所以 ragged 步必须走 CUDA graph，不是可选项；做法见下一小节（补空行，不用分桶）。
- **n-gram 混合负载按序列没赚**：679.4 对 batch 级 675.6、基线 720.2（都是 ×0.94）。接受率从 0.27 升到 0.44（chat 那一半不再白算），ragged 步 48%，但吞吐没动。一部分是 eager（约 4 个点），一部分是错开的探测；第二次作业再看。
- 回归正常：edit batch 32 草稿模型开关按序列 904.0 对 batch 级 908.1（ragged 步 6%，差 0.5%）；两次混合基线 720.2 / 719.7。
- GPU 测试里 `spec_per_sequence` 的贪心用例断言“必须出现 ragged 步”，四个短问题的 n-gram 草稿全被砍到 0，没出现；换成上面说的混合 batch 用例。

### 复测（作业 233732，2026-10-29，代码 b1d90de，ragged 模式 varlen，43 分钟）

脚本 `cluster/nanovllm-perseq2-a40.sbatch`（和第一次一样的行，加 batch 8 的 eager 基线），跑在一台 300 W 的 A40 节点上（基线比 233573 的 250 W 节点高，只在作业内比）。原始行在 `docs/benchmarks/qwen3-8b-a40-perseq.txt`。两处 bug 修了之后，数字把结论说清楚了：**eager 的 ragged 步把按序列的收益全吃掉了**。

- 混合负载 batch 32：基线 846.2；n-gram 开关 batch 级 784.5（×0.93）、按序列 741.9（×0.88，57% 的步是 ragged 步，接受率 0.27 → 0.44）；草稿模型固定 k=4 961.3（×1.14）、开关 batch 级 943.6（×1.12）、按序列 777.5（×0.92，78% ragged，接受率 0.57 → 0.64）。接受率涨了，吞吐反而掉：ragged 步里 target 和草稿模型的 K 次提案、追赶步全走 eager，草稿模型 28 层的小 kernel 没了 CUDA graph 比 target 亏得多（batch 32 上 target 单独 eager 只亏 4%）。
- 混合负载 batch 8：基线 252.3，eager 基线 190.4——**CUDA graph 在 batch 8 上值 25%**（batch 32 上 4%）；草稿模型开关 batch 级 349.1（×1.38）、按序列 282.7（×1.12，73% ragged）。
- chat batch 32：基线 1044.0；n-gram 按序列 952.2（×0.91，21% ragged：探测已经同步，这 21% 是第 1 位接受率留在 0.2 之上的序列跑 k=1、其他跑 0，是真的参差）；草稿模型按序列 948.2（×0.91，72% ragged）。
- edit batch 32 回归正常：草稿模型开关 batch 级 1008.7（×1.46）、按序列 1005.2（×1.46，6% ragged）。两次混合基线 846.2 / 845.1。
- 两行没出数（混合 batch 32 的 eager 基线、chat batch 32 草稿模型开关 batch 级）：进程一行都没打就退了，日志里没有 traceback 也没有 Killed。第三次作业的脚本记每一行的退出码，没出结果行的重跑一次。
- 重编译查过了：日志里 292 次 `Recompiling` 全在计时之前（基线一行 12 次、固定 k=4 17 次、开关 21 次、按序列 23 次，多的两次是 ragged 采样分支 k=1 和 k≥2 的预热；每一行最后一次重编译都是 bench 计时前那条 63 个 token 的预热 prefill）。在 CPU 上也验了一遍：预热之后，varlen 和 pad 两种布局、各种 batch 和草稿数的 ragged 步都不再编译。
- GPU 测试 42/44（8 个跳过）：挂的是 n-gram 两种按序列模式的混合 batch 用例，输出和参考一致，但 `num_ragged_steps == 0`。当时判断是一个样本就把抄写任务的接受率打到 0，在 cc2a84a 加了先验样本；234187 证明不是（见下），原因在做法一节。

结论：varlen 模式下按序列的 k 不能做默认。它该赢的地方（接受率）赢了，输在每个 ragged 步都是 eager；要么 pad 模式把 graph 保住（第三次作业 234187，varlen 对 pad，同一场跑），要么按序列只在 batch 很小、行几乎免费的时候开。

### ragged 步的第二种跑法：补空行、保住 CUDA graph（2026-10-29 加）

`LLM(..., speculative_ragged="pad")`（作业 234187 之后是默认值；`"varlen"` 留着）。上面 batch 8 的教训是 ragged 步不能 eager。原来想的是按总行数分桶给 varlen kernel 抓 CUDA graph，但有个更省事的做法，就是草稿模型追赶步已经在用的那一招：**所有序列都算 k_max+1 行，草稿少的序列在前面补空行，自己的行靠右对齐**。`flash_attn_with_kvcache` 的因果掩码按右下角对齐（最后一个 query 对准 `cache_seqlens-1`），所以序列 i 的真实行 j 正好看到 `L_i-1+j` 之前的 key，和它单独算一模一样；前面的空行看到的 key 更少，输出是垃圾，但 slot 置 -1 不写 KV、位置夹到 0，采样器按 `token_index` 只取真实行。于是 ragged 步和统一形状的步走同一条路：同一个 kernel、同一套按 q 抓的 CUDA graph、同一个采样器分支（`token_index` 换成 padded 布局），`propose` 一行不改（草稿少的序列的多余提案写进它自己的空行）。

代价是多算的空行：batch 32 混合负载里 16 条 k=4、16 条 k=0 的步，varlen 算 96 行，pad 算 160 行；n-gram 在 chat batch 32 上 k=4（160 行、全拒）比不投机慢 35%（作业 225010），所以 batch 大时空行不是免费的，而 varlen 的 eager 只亏 4%。batch 8 时反过来：行几乎免费，eager 亏 11%。两种模式都留着，第三次作业（234187）在 batch 8 和 32 上各测一遍再定默认值，结果在下面。

- `ModelRunner.ragged_layout(num_drafts, pad=True)`：每条序列 `1+k` 行，`token_index` 指向靠右的真实行，`write_index` 里没有的草稿指向本序列的空行（不再需要 scratch 行）。
- n-gram 路径：按序列先放 `k - k_i` 个空行（token 0、位置 0、slot -1）再放真实行；context 和统一步一样。草稿模型路径：位置 `L-1+c-front` 夹到 0，slot 按 `c >= front` 屏蔽，最后一个 token 放到第 `front` 列；追赶步不变。
- 模拟器：`_perseq` 模式里 varlen 和 pad 随机各半；pad 步必须没有 `cu_seqlens_q`（走 graph 那条路）、布局和 `ragged_layout(pad=True)` 一致、空行 slot 都是 -1、真实行的位置/slot/KV 检查照旧，采样器用 padded 布局。注入 5 种 bug（空行不屏蔽 slot、输入 token 放错列、n-gram 空行写 KV、多余草稿写进别的序列的行、pad 步仍设 varlen context），30 个种子里各有 17–18 个用例抓到。
- GPU 测试加 `spec_per_sequence_pad`、`async_draft_per_sequence_pad`（pad 成默认之后改名为 `_varlen` 两种）。

### 实测（作业 234187，2026-10-29，代码 cc2a84a，49 分钟）：varlen 对 pad，和按序列 k 的结论

脚本 `cluster/nanovllm-perseq3-a40.sbatch`（第一次提交 233916 和 233732 同一秒启动，抢同一个 checkout 的 `.git/index.lock` 死了，脚本加了重试再交），19 行全部出数（脚本记每行退出码、没结果行就重跑一次，这次没用上）。同一场里每个负载/batch 先跑基线，原始行在 `docs/benchmarks/qwen3-8b-a40-perseq.txt`：

- 混合 batch 8：基线 251.5；草稿模型开关 batch 级 347.6（×1.38）；按序列 varlen 263.3（×1.05，78% ragged）、**pad 312.8（×1.24）**；n-gram 按序列 varlen 207.8（×0.83）、pad 242.6（×0.96）。
- 混合 batch 32：基线 844.5；草稿模型开关 batch 级 945.9（×1.12）；按序列 varlen 862.4（×1.02）、pad 839.4（×0.99）；n-gram 开关 batch 级 780.5（×0.92）、按序列 varlen 740.8（×0.88）、pad 725.4（×0.86）。
- chat batch 32：基线 1038.0；n-gram 按序列 varlen 882.1（×0.85，50% ragged）、pad 984.2（×0.95）；草稿模型按序列 varlen 1005.1（×0.97）、pad 1059.3（×1.02）。
- 漂移：混合 batch 8 基线 251.5 / 252.3。

两个结论：

1. **pad 对 varlen**：batch 8 上 pad 赢 19%（草稿模型）和 17%（n-gram），chat batch 32 上赢 5% 和 12%；只有混合 batch 32 上 varlen 赢 2–3%（96 行对 160 行，这个 batch 下行不再免费）。默认改成 `"pad"`。
2. **按序列 k 本身，在这台 A40、8B + 0.6B 的组合上，哪一格都没赢过 batch 级开关**：混合 batch 8 ×1.24 对 ×1.38，batch 32 ×1.02 对 ×1.12，chat batch 32 草稿模型 ×1.02（batch 级在 233573 里 ×1.20）。接受率确实涨了（草稿模型 0.57 → 0.63，n-gram 0.27 → 0.43），输出逐 token 和参考一致，机制是对的，输在两种 ragged 步各有一笔省不回来的账：varlen 省了行，丢了 CUDA graph（batch 8 上值 25%），草稿模型的 K 步提案和追赶也跟着 eager；pad 保住了 graph，但每条序列照样算 1+k_max 行、草稿模型照样跑 k_max 步，省下的只是被砍掉的那几位的验证结果——pad 模式下"按序列砍 k"和"整个 batch 用 k_max"算的一样多，只是少接受几个 token（batch 8 上平均每步 3.1 个草稿对 3.8 个，×0.93 的账和实测 ×0.90 对得上），必然不如 batch 级。
   更根本的是上限本来就小：batch 32 上固定 k=4 ×1.14、batch 级开关 ×1.12，说明混合负载里 chat 一半"白算"的位置只花了两个点——decode 在这个 batch 下仍是读权重为主，多几行接近免费；按序列 k 能省的最多就是这两个点，而 ragged 步的代价远不止。预期那一节没把这笔账算清楚。要赢需要两件事一起成立：ragged 注意力进 CUDA graph（按总行数分桶，输入 pad 到桶），草稿模型只给需要的序列跑需要的步数而不是 batch 的 k_max；而且只在 target 算力受限、行不再免费的 batch 下才有赚头。两项都没做，模式留着（默认关），数据留在这里。

- GPU 测试 42/44（8 个跳过），和 233732 一样挂在 n-gram 两种按序列模式的混合 batch 用例（`num_ragged_steps == 0`，输出和参考一致），说明 cc2a84a 的先验不是它的原因；原因见做法一节（测试的 prompt 开着思考模式，抄写任务没开始抄），用例已改，作业 234580 验证通过（44/44）。
- 重编译 345 次，比 233732 的 292 次多的是 pad 行的预热；都在计时之前。

## 温度采样下的无损投机解码：草稿按分布采样（2026-11-03 加）

`LLM(..., speculative_model=..., speculative_sampled_drafts=True)`（有草稿模型时默认开；`False` 回到 argmax 草稿）。之前草稿模型的提案一直是 argmax，q 是 one-hot，拒绝采样按 `p(d)` 接受、被拒后从"p 去掉 d"里重采——这在贪心下是精确的，温度采样下也精确（one-hot 是 `min(1, p/q)` 的特例，见面试一节），但接受率只有 `p(argmax)`，也就是 target 对自己最可能的 token 的信心，温度一高就掉。标准做法（Leviathan 等 2023）是草稿从草稿模型自己的分布 q 里采样，验证按 `min(1, p/q)` 接受，被拒的那一位从残差 `norm(max(0, p - q))` 里重采；输出分布严格等于 target 自己采样的分布，接受率是 `Σ min(p, q)`，q 越像 p 越高——草稿模型给自己打草稿时接近 1。

### 做法

- **采样草稿（`DraftSampler`，`ModelRunner.propose` 里）**：草稿模型每一步的 logits 按序列各自的温度采一个 token，Gumbel-max 直接做在 logits 上（`argmax(logits / T − log Exp(1))`，和先算 softmax 再 `probs / Exp(1)` 取 argmax 是同一个分布，少两遍 (B, V) 的读写）；温度 0 的序列仍取 argmax。存下来的不是 q 而是这一步的 logits 本身，bf16、形状 `(B, k, V)`（batch 32、k=4、Qwen3 的 15 万词表是 39 MB，每步分配一次），加上每行的 log-sum-exp（`draft_log_norms`，`(B, k)`，和 argmax 在同一个编译的 kernel 里顺带算出来），q 在采样器里按 `exp(logit/T − log_norm)` 读。采样在 CUDA graph 之外（graph 只抓草稿模型的 forward），所以不碰 graph 里的随机数状态；`DraftSampler` 和两个采样器一样 `torch.compile(dynamic=True)`，在 `warmup_samplers` 里按 batch 8 和 1 预热。三个采样器里所有对整个词表的 argmax / log-sum-exp 都分两级做（`chunked_argmax` / `chunked_logsumexp`：先把一行切成 128 等份各取 max 和下标，再在 128 个块里取），噪声用 `torch.rand` 在 kernel 里现生成（`exponential_like`），原因见复测 2（作业 278958）。（第一版 9b5aa91 是 eager 的 `sample_drafts`：float32 的 softmax、噪声、相除、argmax，q 存成 float32 的 `draft_probs`；A40 上量出每步 2–3 ms，见实测，2026-11-04 改成现在这样。）TP 下只在 rank 0 采，token 照旧广播，q 只有 rank 0 要。整个 batch 都是贪心（T=0）时 `run`/`run_async` 不把温度传给 `propose`（`draft_temperatures`），这一步就不算 softmax 也不存 q：贪心行的草稿本来就是 argmax，采样器验证贪心行也不读 q，结果一个 token 都不变，只省掉作业 244661 里量到的每步 2 ms（batch 32 下 3%）。
- **验证（`RejectionSampler`）**：多两个可选参数 `draft_logits`、`draft_log_norms`。`q_j(d_j)` 从草稿 logits 读：`log q = logit_j[d_j]/T − log_norm_j`，接受概率 `exp(log p − log q)`（和 `rand` 比，大于 1 就是必收），都在编译的图里，对 (B, k, V) 只有点读没有归约（第一版在采样器里对草稿 logits 做 logsumexp，k=1 的那张图在 GPU 上触发了 Inductor 的融合断言 `simd.py:1223`，作业 245130，见复测）；被拒那一位的残差 `max(0, p − q_j)`，`q_j = exp(logits_j/T − log_norm_j)` 只对那一行算，`torch.where` 按"真的拒了"（不是全收、也不是 ragged 的补位）选残差还是 p；残差全零（p 和 q 完全相同，理论上不会拒）时回退到 p。贪心行不看 q，照旧只收 argmax。`draft_probs=None` 走原来的 one-hot 路径（n-gram、argmax 草稿）。ragged batch 的 `token_index` 分支一样接 q：q 按序列和位，补位不读。
- **异步调度**：q 和草稿在同一步里产生、同一步里消费（`run_async` 里 `propose` 之后就是 `sample`），不跨步，所以在途 token 的机制一个字不改。
- 预热：`warmup_samplers` 对 `draft_logits` 分支（统一和 ragged）各编一次，`DraftSampler` 按 batch 8 和 1 编一次；bench 加 `--greedy-drafts`，行里多一个 `sampled_drafts=` 字段。CPU 上照 233732 的方法验过：预热之后，batch 1–32、k 1–4、三种温度、五种 ragged 布局的草稿采样和验证都不再编译（12 张图，预热前后不变）。

### 正确性验证（CPU）

- `tests/test_sampler.py`：`DraftSampler` 的频率等于 `softmax(logits/T)`、贪心行取 argmax、近 one-hot 的行必取峰值；**核心用例**：三个位置各一个 target 分布和一个不同的草稿分布（词表 6，T ∈ {0.7, 1, 1.5}），10 万个 batch，草稿从 q 采，跑真实的采样器——每个位置上输出 token 的频率等于 target 分布（误差 < 0.012），第 j 位的接受率等于 `Σ min(p_j, q_j)`，被拒之后的 token 从不等于被拒的草稿；one-hot 的 q（logits 峰值 0、其余 −1e4，float32 下 softmax 严格 one-hot）和 `None` 路径在相同随机数下收同样的草稿、重采的 token 99.9% 相同；贪心行传不传 q 结果逐位相同；草稿 logits 用 bf16 传（GPU 上的真实情形）输出仍是 target 分布。
- `tests/test_ragged_sampler.py`：ragged batch（一条两个草稿、一条没有）上同样的分布检验。
- 模拟器：`_propose` 把序列温度传给真实的 `propose`（整个 batch 贪心时像 `run` 一样传 `None`，并核对这时没有存 logits），核对 `draft_logits` 的形状 `(B, k_max, V)` 和每行的峰值等于那条序列的草稿（假草稿模型给自己的 token 一个确定的 one-hot logits，采样结果和 argmax 一致，所以其余检查原样有效）；采样器调用带上 q，随机用例里 70% 开采样。所有既有检查（KV、哈希块、输出等于参考）照旧。
- 变异（8 种，分布检验全抓到 7 种）：

| 注入的 bug | 结果 |
|---|---|
| 按 p 接受而不是 p/q | 抓到 |
| 残差不 clamp（p−q 有负数） | 抓到 |
| 残差用 p（不减 q） | 抓到 |
| 残差用前一位的 q | 抓到 |
| 全收之后也用残差 | 抓到 |
| 残差不归一化 | 等价变异：Gumbel 采样对缩放不敏感，不是 bug |
| 草稿采样不除温度 | 抓到（`sample_drafts` 的频率用例） |
| q 按温度算、草稿却取 argmax | 抓到 |

q 收进采样器之后（2026-11-04，log-sum-exp 由 `DraftSampler` 算的版本）再注入 12 种，对 7 个分布用例（`test_sampler` 的三个温度、bf16 草稿 logits、`DraftSampler` 频率和 log-sum-exp、one-hot 对照，`test_ragged_sampler` 的 ragged 分布）：

| 注入的 bug | 失败的用例数（共 7） |
|---|---|
| 按 p 接受而不是 p/q | 5 |
| `DraftSampler` 的 log-sum-exp 不除温度 | 1（它自己的用例；采样器用例自己算归一化项，看不到） |
| q 在草稿处的 logit 不除温度 | 2（T=1 的用例看不出，T=0.7 / 1.5 抓到） |
| 残差不 clamp | 2 |
| 残差用 p（不减 q） | 6 |
| 残差用前一位的 q | 6 |
| 残差那一行的 q 不减归一化项 | 5 |
| 残差那一行的 q 不除温度 | 2 |
| 全收之后也用残差 | 6 |
| 草稿采样不除温度 | 1（`DraftSampler` 的频率用例） |
| 草稿采样用均匀噪声代替 Gumbel | 1 |
| 草稿采样返回 argmax | 1 |

### 预期

- chat T=0.8、Qwen3-8B + 0.6B 草稿：接受率从 `p(argmax)`（0.6B 在 chat 上的信心，估计 0.4–0.5）升到 `Σ min(p, q)`（估计 0.6 以上），batch 1 的 TPOT 相应下降；edit 上 argmax 本来就接近必中，差别小。
- T=0 一个 token 都不该变（贪心行走原路径）：作业里放一行 T=0 的 chat 对照。
- 代价：每步一个 `(B, k, V)` 的 float32 softmax 和一次 gather，batch 32 时几十 MB 的写读，预计 1% 以内。

### 实测（作业 244661，2026-11-03，代码 9b5aa91，56 分钟）

脚本 `cluster/nanovllm-sampled-a40.sbatch`：先跑 GPU 测试（14 种模式，新加 `draft_greedy_drafts` 模式和一个接受率用例），再跑 23 行 bench，每个格子都是同一场作业里"基线、argmax 草稿（`--greedy-drafts`）、采样草稿"三行连着跑。原始行在 `docs/benchmarks/qwen3-8b-a40-sampled.txt`。GPU 测试 50 过、20 跳：0.6B 给自己打草稿、T=1 时采样草稿的接受率 > 0.9（`draft`、`async_draft` 两种模式），argmax 草稿只有 0.2–0.95（`draft_greedy_drafts`）——就是上面说的"q 越像 p 接受率越高"，在 GPU 上验到了。吞吐（tok/s，括号里是对基线的倍数和接受率）：

| 负载，T=0.8 | batch | 基线 | argmax 草稿 k=4 | 采样草稿 k=4 | 采样 / argmax |
|---|---|---|---|---|---|
| chat | 1 | 38.8 | 65.2（×1.68，0.389） | 67.4（×1.74，0.418） | +3% |
| chat | 8 | 288.9 | 369.0（×1.28，0.399） | 393.2（×1.36，0.426） | +7% |
| chat | 32 | 1036.8 | 1026.0（×0.99，0.396） | 1073.4（×1.04，0.421） | +5% |
| chat，开关 0.2 | 32 | 1036.8 | 1145.6（×1.10，0.446） | 1163.5（×1.12，0.459） | +2% |
| edit | 1 | 38.4 | 108.2（×2.82，0.868） | 107.9（×2.81，0.871） | 0 |
| edit | 32 | 692.5 | 1030.9（×1.49，0.843） | 990.9（×1.43，0.845） | −4% |
| 混合 | 32 | 849.7 | 876.4（×1.03，0.552） | 894.8（×1.05，0.561） | +2% |
| chat，T=0 对照 | 32 | — | 1119.5（0.407） | 1085.0（0.407） | −3% |

- **接受率涨了，但没有预期的那么多**：chat 上每一位都涨 0.02–0.04（第 1 位 0.63 → 0.66，第 4 位 0.21 → 0.23），整体 0.39 → 0.42，不是预期的 0.6。预期算错在把 `Σ min(p, q)` 当成了"q 像 p"，0.6B 和 8B 在开放问答上的分布差得远，重合只比 argmax 的信心多这么一点。按位接受率加起来，每步平均多收的 token 从 1.58 升到 1.69（+7%），步数少 9%（520 → 474）。
- **代价比预期大**：batch 32 上每步从 58.2 ms 涨到 61.0 ms（+2.8 ms，5%），batch 8 +1.0 ms，batch 1 +0.4 ms；T=0 对照行接受率、token 数、步数三者完全相同（1 个 token 都没变，贪心路径验证成立），但也多付 1.9 ms。来源就是做法里那个 `(B, k, V)` float32 的 q：`sample_drafts` 在 CUDA graph 之外，对 batch 32 的 `(32, 151936)` 做 float 转换、除温度、softmax、指数噪声、相除、argmax，再拷进 `draft_probs`，每个草稿步约 330 MB 的读写，4 步加起来 1.3 GB，A40 的带宽下就是 2 ms。预期里写的"1% 以内"只算了存 q，没算这一串 eager 算子。所以 chat batch 32 上 +7% 的 token 只落成 +5% 的吞吐，batch 1 上 +3%；edit 上接受率本来 0.87，采样没得涨（0.868 → 0.871），batch 1 持平，batch 32 的 −4% 不是每步变慢（97.4 对 97.1 ms），是采样出来的回答短了（25015 对 25394 个 token，253 对 260 步），在采样的噪声里。
- 开关 0.2 下采样草稿把 k 停在更高的位（k=4 的步 145 对 69），接受率 0.446 → 0.459，吞吐 +2%。
- 接下来做两件事：T=0 的整 batch 跳过采样（已做，见做法）；把 q 的那一串算子收进编译的采样器——草稿步只存 bf16 的 logits（一半的字节），用 Gumbel-max 直接在 logits 上采（`argmax(logits/T + Gumbel)`，不用先算 softmax），`p/q` 和残差要的 q 在 `RejectionSampler` 里从 logits 算（和 target 的 p 一样的 log-sum-exp，编译后和现有算子融合）。预计把 2.8 ms 压到 0.5 ms 以下。2026-11-04 做了（做法一节已改成新写法），CPU 上分布检验照旧全过、11 种变异见下表，A40 复测脚本 `cluster/nanovllm-sampled2-a40.sbatch` 跑和 244661 一样的 23 行。

### 复测（作业 245130，2026-11-04，代码 9b23ec9，q 收进采样器的第一版）

两个结果，都不是想要的：

- **k=1 的采样器图在 GPU 上编译崩了**：`torch._inductor.exc.InductorError: AssertionError` 在 `simd.py:1223`（`assert rnumel1 == 1 and rnumel2 != 1`，把一个非归约节点和一个归约节点垂直融合时的断言）。固定 k=4 只编 q=5 的图，没事；开关模式预热 q=2..5，q=2（k=1，`torch.compile` 把 1 特化成常数）那张图崩：GPU 测试的 `draft_acceptance` 模式 5 个用例在建引擎时报错（其余 47 过、18 跳），bench 里 chat batch 32 采样草稿 + 开关 0.2 那一行两次都没出数。CPU 上同样的预热（C++ 后端）不崩，所以 CPU 的零重编译检查抓不到它。新加的归约是对 `(B, k, V)` 草稿 logits 的 logsumexp，k=1 时和 target 那边同形状的 logsumexp、`cumprod(dim=1).sum(dim=1)` 这类 size-1 归约混在一张图里；没有在 GPU 上逐个排查（一次作业排几小时队），改成绕开：log-sum-exp 由 `DraftSampler` 算（它本来就把 logits/T 整行读了一遍，多一个归约在同一个 kernel 里），采样器拿到 `(B, k)` 的 `draft_log_norms`，对草稿 logits 只剩点读和指数，没有归约（做法一节已更新）。
- **每步开销没降**：同一场作业内，chat batch 32 argmax 草稿 59.0 ms/步，采样草稿 61.4（244661：58.2 对 61.0）；batch 8 44.2 对 45.5（244661：44.3 对 45.3）；batch 1 39.1 对 39.7。也就是说 244661 里归到"eager 的 softmax 和 float32 的 q"头上的 2–3 ms，去掉这些之后还在。按带宽算，编译后的 `DraftSampler` 四次调用加 bf16 logits 的拷贝不到 1 ms，所以这 2 ms 另有来源（候选：编译函数每次调用的 guard 和启动开销、`exponential_` 在 Inductor 里是否退回 aten、采样器 `draft_logits` 分支的 kernel 数）。猜不如量：bench 加 `--profile N`（预热后对一个 batch 跑 `torch.profiler`，按 CUDA 时间和 CPU 时间各打前 N 个算子），第三次作业 `cluster/nanovllm-sampled3-a40.sbatch` 先跑 GPU 测试（`draft_acceptance` 必须过），再把 chat batch 32 的 argmax 和采样两行各 profile 一次，然后跑对比用的行。
- 跑出来的 16 行里接受率和 244661 一致（chat 采样 0.41–0.42、argmax 0.39–0.40；edit 0.84–0.85），吞吐 chat batch 1 67.4 对 65.8、batch 8 367.8 对 362.0、batch 32 1061.5 对 1033.7，edit batch 32 959.5 对 1024.4（又是采样回答偏短：264 步对 242 步，每步 98.4 对 100.8 ms）。
- 重编译：日志 382 次，基线行 12、argmax 草稿行 17、采样草稿行 19（多的 2 次是 `draft_probs` 分支在 batch 8 和 1 的预热）、开关行 21 / 25，都在计时之前。这次的判断方法比 233732 更硬：每一行最后一次重编译的时间加上这一行的计时时长，再加 19–21 秒（下一个进程启动和读权重），正好是下一行第一次重编译的时间——22 对相邻行里 21 对都是这样（唯一的例外是开关采样行到 edit 基线行，多了 12 秒，来源没查），所以计时循环就从最后一次重编译（bench 计时前那条单 prompt 的预热）之后开始。

### 复测 2（作业 278958，2026-11-08，代码 a5c238c，profiler）

脚本 `cluster/nanovllm-sampled3-a40.sbatch`，52 分钟，250 W 档的 A40（绝对数比前几次 300 W 的节点低 10% 左右，只在作业内比）。原始行在 `docs/benchmarks/qwen3-8b-a40-sampled.txt` 第二段。

- **崩溃修好了**：GPU 测试 50 过、20 跳、0 错，`draft_acceptance` 模式（k=1 的采样器图）正常；chat batch 32 采样草稿 + 开关 0.2 那一行也出数了。
- **同场对比，采样草稿全面不慢于 argmax 草稿**（tok/s，括号里是接受率）：chat batch 32 基线 874.4，argmax 898.2 / 934.0（两次，0.390 / 0.396），采样 974.2 / 969.5（0.420 / 0.416），+4% 到 +8%；开关 0.2 下 1001.3 对 1002.5；batch 8 334.8 对 341.0（+2%）；batch 1 58.4 对 59.5（+2%）；混合 batch 32 780.8 对 778.8（持平，接受率 0.548 → 0.563）；**T=0 对照 996.9 对 997.2，token 数和步数完全相同**——贪心 batch 跳过采样起效了，244661 里 −3% 的那笔没了。
- **每步开销还在**：chat batch 32 argmax 草稿 66.5 ms/步（两次都是），采样 68.3 / 68.6（+2.0 ms）；batch 8 50.1 对 51.3（+1.2）；batch 1 44.3 对 45.0（+0.7）。
- **profiler 找到了去处**（预热后对一个 batch 32 跑 `torch.profiler`，约 240 步）。argmax 那一行里采样相关的 Triton kernel 共 41 ms（主采样器一个 kernel 167 µs/步）；采样那一行里 359 ms：`DraftSampler` 的 kernel 183 µs × 972 次（每步 4 次）= 178 ms，拒绝采样器 `draft_logits` 分支的两个 kernel 341 + 334 µs/步 = 161 ms，再加一个 86 µs/步的 kernel。折合每步 1.5 ms 对 0.17 ms，差 1.3 ms；剩下的 0.7 ms 是每步 4 次 9.7 MB 的 logits 拷贝（存进 `draft_logits`）和 4 次编译函数调用的 CPU 开销。
- **为什么一个 32 行 × 15 万列的 argmax 要 183 µs**：读 9.7 MB 的 bf16，按 A40 的带宽只要 15 µs。答案在 Inductor 源码里（`torch/_inductor/ir.py`，`Reduction.num_splits`）：`should_split = ... and reduction_type not in arg_reduction_types`——带 argmax / argmin 的归约**从不**做 split reduction（不切成多个线程块再合并），一行就是一个线程块。batch 32 就是 32 个块，A40 有 84 个 SM，一大半闲着；batch 1 整个 GPU 只有 1 个块在干活。而 argmax 和它融合在同一个 kernel 里的 log-sum-exp、噪声、除法也跟着只用 32 个块。主采样器（167 µs）同样中招，只是每步只跑一次所以一直没被注意。另一个小项：`exponential_` 在 Inductor 里是 fallback（`lowering.py`：`make_fallback(aten.exponential.default)`），噪声先由 aten kernel 写进一个 19 MB 的 float32 张量再被读回来，而 `torch.rand` 是在消费它的 kernel 里直接生成的。
- **改法（e88f54d）**：`chunked_argmax(x)` 把最后一维 view 成 `(…, 128, V/128)`（128 取能整除词表的最大 2 的幂，Qwen3 的 151936 = 2^7 × 1187 正好 128），每块 `max(dim=-1)` 得到值和块内下标，再对 128 个块值取 argmax、加上块偏移——结果和 `torch.argmax` 逐位相同（并列取第一个，两级都如此），第一级有 `rows × 128` 个输出可以铺满 GPU；`chunked_logsumexp` 同样两级。三个采样器所有对词表的归约都换成这两个；噪声改成 `-log(rand)`；主采样器也顺手改成在 logits 上做 Gumbel-max（不再先算 softmax）。CPU：2157 个测试全过，新加的用例核对 6 种词表大小（含奇数、单块退化）、并列、−inf 行、列切片下两级归约和原版逐位相同，`exponential_like` 的均值和方差是 1；再注入 19 种 bug（上表 12 种改写到新代码，加上块偏移忘加 / 偏移乘错 / 取最小块 / 第二级用 max 代替 log-sum-exp / 噪声不取对数 / 残差除以 log E / 贪心行比错行）全部抓到；预热后零重编译照旧。第四次作业 `cluster/nanovllm-sampled4-a40.sbatch`：GPU 测试，`bench_sampler.py`（单独计时三个采样器每次调用的微秒数，新旧 `sampler.py` 各跑一遍），再跑 278958 的 17 行。
- 顺带记一条 CPU 重编译检查暴露的旧问题：`RejectionSampler.forward` 的变体（统一 / ragged × 有无 `draft_logits` × batch 8 / 1）在 CPU 脚本里是 9 个，超过 `torch._dynamo` 默认的 `recompile_limit=8`，第 9 张图退回 eager。真实的 `warmup_samplers` 不开 `speculative_per_sequence` 时只有 4 个变体，开了会撞上；a5c238c 之前就这样，还没处理。

### 复测 3（作业 279567，2026-11-09，代码 a7e5957 = e88f54d 的采样器，49 分钟，300 W 节点）

脚本 `cluster/nanovllm-sampled4-a40.sbatch`：GPU 测试（50 过、20 跳），`bench_sampler.py` 单独计时每个采样器每次调用的微秒数（新旧 `sampler.py` 各跑一遍），再跑 278958 的 17 行。原始行在 `docs/benchmarks/qwen3-8b-a40-sampled.txt` 第三段，微基准在 `results/279567/bench_sampler.txt`。

| 每次调用 µs | batch 1 旧 → 新 | batch 8 旧 → 新 | batch 32 旧 → 新 |
|---|---|---|---|
| `Sampler` | 172 → 118 | 167 → 138 | 201 → 188 |
| `DraftSampler` | 224 → 137 | 186 → 156 | 219 → 220 |
| `RejectionSampler`，one-hot 草稿 | 296 → 351 | 318 → 367 | 584 → 794 |
| `RejectionSampler`，采样草稿 | 320 → 345 | 728 → 493 | 994 → 1045 |
| 参照：eager `argmax` / `logsumexp` | 12 / 83 | 12 / 84 | 30 / 320 |

- **单行的判断对了，batch 32 的没对**。batch 1 上 `Sampler` 和 `DraftSampler` 快了 31% / 39%：一个线程块串行扫 15 万个元素的那条延迟链确实没了。但 batch 32 上 `DraftSampler` 原地不动（219 → 220），拒绝采样器反而慢了 36%。端到端：chat batch 32 采样草稿比 argmax 草稿每步多 0.8–1.5 ms（278958 里是 2.0），batch 8 多 0.5，batch 1 持平；吞吐采样 / argmax 为 +2.0% / +3.6%（两次）、开关 −0.7%、batch 8 +3.6%、batch 1 +7.5%（这次接受率碰巧 0.448）、混合 +1.7%；T=0 对照 token 和步数仍完全相同。
- **profiler 说明了为什么**（采样行里 `DraftSampler` 每次调用的 5 个 kernel）：`triton_poi_fused_rand_like` 24.5 µs——噪声还是先写成 19 MB 的张量：Inductor 的 `inductor_random` 对每个随机张量都 `realize()`（`lowering.py`），噪声根本不可能融进消费它的 kernel，`exponential_` 换 `rand` 白换；`triton_red_fused_logsumexp_max` 105 µs——第一级归约（greedy 的 max+下标和 log-sum-exp）是 **looped** 归约：块宽 1187 超过 Inductor persistent 归约的上限 1024，log-sum-exp 的 max 和 sum 两遍各读一次 logits；`triton_red_fused_max` 52 µs——Gumbel 那一支再读一遍 logits 加 19 MB 噪声；两个第二级 kernel 各 2 µs。内存流量从旧版的"读一遍 logits + 噪声写读各一次"涨到了三遍 logits，正好抵消了并行度的收益。拒绝采样器里对 `(B·k, V)` 的 log-sum-exp 本来 Inductor 自己会切分（它不带 argmax），分块反而让它和 greedy 的 argmax 融成了一个两遍读 39 MB 的 looped kernel（384–400 µs/步，之前没有这么大的一项）。
- **改法（下一版）**：块缩到 ≤1024 个元素——`_chunks` 取能整除词表的、不超过 1024 的最大 2 的幂做块宽（151936 → 128 宽、1187 块），这样第一级是 persistent 归约，一块留在寄存器里，max、下标、log-sum-exp 的 max 和 sum、Gumbel 的 max 都从同一次读取里出，再加 `bench_sampler.py --profile`（每个算子的 CUDA kernel、每次调用的启动数和微秒数），第五次作业 `cluster/nanovllm-sampled5-a40.sbatch` 三个版本（新、e88f54d、a5c238c）同场比。
- 每步剩下的开销里还有一项不在 GPU 上：batch 1 的 `DraftSampler` 137 µs 里 GPU 只干了几十微秒，其余是每次编译函数调用的 CPU 开销（Dynamo 守卫 + 5 个 kernel 的启动）；一步调 4 次就是 0.4–0.5 ms，只有把采样器放进 CUDA graph 才能省掉，而那要求形状固定。

### 复测 4（作业 281207，2026-11-09，代码 e1174a6 = 62ddc96 的采样器，50 分钟，300 W 节点）

脚本 `cluster/nanovllm-sampled5-a40.sbatch`：GPU 测试（50 过、20 跳），`bench_sampler.py --profile` 在 HEAD 上跑（每个算子的 CUDA kernel 和 Inductor 生成的 Triton 代码都留了下来，`results/281207/inductor_code.txt`），再用 `git show` 换回 e88f54d 和 a5c238c 的 `sampler.py` 各跑一遍（同一场、同一张卡），最后是 278958 的 17 行。原始行在 `docs/benchmarks/qwen3-8b-a40-sampled.txt` 第四段。

| 每次调用 µs（batch 32） | 62ddc96（128 宽块） | e88f54d（1187 宽块） | a5c238c（不分块） |
|---|---|---|---|
| `Sampler` | **166** | 187 | 196 |
| `DraftSampler` | 213 | 219 | 218 |
| `RejectionSampler`，one-hot 草稿 | **469** | 793 | 582 |
| `RejectionSampler`，采样草稿 | **733** | 1044 | 993 |

batch 8 四项依次 157 / 214 / 395 / 429（e88f54d 136 / 151 / 331 / 492），batch 1 130 / 151 / 389 / 425（e88f54d 126 / 132 / 320 / 337）：小 batch 的墙钟时间是编译函数每次调用的 CPU 开销（batch 1 的 `DraftSampler` 151 µs 里 GPU 只有 17 µs），kernel 越多越慢，所以 62ddc96 在 batch 1/8 上反而比 e88f54d 慢一点。

- **结论**：128 宽的块是三个 torch 版本里最好的——batch 32 上 `Sampler` −15%、拒绝采样器 −19% / −26%——但 `DraftSampler`，也就是一步要调 4 次的那个，三个版本都是 213–220 µs，一动不动。
- **profiler 给出了原因**（62ddc96 的 `DraftSampler`，batch 32，6 个 kernel，GPU 共 207 µs）：`triton_red_fused_logsumexp_max` 124 µs——greedy 的 max+下标和 log-sum-exp 这一级 Inductor 仍然编成了 **looped** 归约（78 GB/s，块明明只有 128 个元素），而同一个函数里 Gumbel 那一级 `triton_per_fused_max` 是 persistent 的，42.6 µs 读 29 MB（约 680 GB/s，接近带宽）；`Sampler` 里正好反过来（Gumbel 级 looped，85 µs）。`triton_poi_fused_rand_like` 29 µs：噪声在三个版本里都是先写成 19 MB 的张量再读回。也就是说，从 PyTorch 图这一侧既控制不了 Inductor 对 persistent / looped 的选择，也去不掉噪声张量；`TORCHINDUCTOR_COOPERATIVE_REDUCTIONS=1` 还没试，但它改的是单行归约怎么跨块协作，噪声那 19 MB 照样在。
- **端到端**：chat batch 32 每步 argmax 58.8 / 59.5 ms、采样 60.4 / 60.1 ms（多 0.6–1.6；279567 是 0.8–1.5，278958 是 2.0），batch 8 44.0 对 44.4，batch 1 38.9 对 39.0。吞吐采样 / argmax：chat batch 32 +5.6% / +1.8%（两次）、开关 −1.3%、batch 8 +10%（接受率 0.435）、batch 1 +4%、混合 +2.9%；T=0 对照 1113.2 对 1115.1 tok/s，token 和步数完全相同。剩下的差距约每步 1 ms，是 60 ms 一步的 2%。

### 手写 Triton 融合采样核（2026-11-09）

`DraftSampler` 和 `Sampler` 在 GPU 上不再走 `torch.compile`，改成 `nanovllm/layers/sampler.py` 里的 `fused_sample`，两个手写的 Triton kernel：

- **第一级** `_sample_tiles_kernel`，grid 为 (行数, 词表 / 1024 向上取整)，每个 program 读一行的一个 1024 宽的片进寄存器（bf16 转 f32，片尾不足的位置补 −inf），从这一次读取里出 6 个数：片内 max 和它的下标（greedy），`logits / T` 的 max 和 `Σ exp(· − max)`（log-sum-exp 拆成两个数），以及 `logits / T + Gumbel` 的 max 和下标（采样）。Gumbel 噪声 `−log(−log U)` 的 U 由 `tl.rand(seed, 行 × 词表 + 列)` 在寄存器里按位置生成，不落内存；seed 每次调用从 torch 的 CPU 生成器取一个，所以 `torch.manual_seed` 仍然能固定采样结果。下标存的时候加上片的偏移。Qwen3 的 151936 列是 149 片，batch 32 共 4768 个 program，跨内存的只有 32 × 149 × 6 个数。
- **第二级** `_sample_rows_kernel`，一行一个 program：在 149 个片的 greedy max 里取第一个最大的片（和 `torch.argmax` 一样，并列取第一个下标），读它的下标；log-sum-exp 由各片的 (max, sum) 合并：`M + log Σ sum_i · exp(max_i − M)`；在各片的 Gumbel max 里取最大的片读它的下标；温度为 0 的行取 greedy。输出 int64 的 token 和 f32 的 log-norm，和 torch 版接口相同。
- 内存流量从"读 logits 两到三遍加写读 19 MB 噪声"降到读 logits 一遍（batch 32 共 9.7 MB），按 A40 的带宽是 20 µs 量级外加两次启动，对比 torch 版的 207 µs；每步 4 次 `DraftSampler` 加 1 次 `Sampler` 的那 1 ms 应该能省掉大半。`RejectionSampler` 这次不动（它的 (B·k, V) log-sum-exp、残差和采样是更多级的归约，等这两个核量过再说）。
- CPU 上没有 GPU 也能验：Triton 自带解释器（`TRITON_INTERPRET=1`，`tests/conftest.py` 在没有 CUDA 时设上，要在 `triton.language` 导入前），kernel 逐 program 用 numpy 跑，约 10 ms 一个 program，所以测试把行数 × 片数控制在几百。`tests/test_triton_sampler.py` 13 个用例：小片宽（8/16）下多片一行，greedy 和 `torch.argmax` 逐位相同（片内并列、跨片并列、−inf 掩码、bf16），log-norm 和 `torch.logsumexp` 一致，全 −inf 的行给 token 0 和 −inf；等 logits 的行采样均匀、尖峰行的采样频率和 softmax 在 4.5 个标准误内；`manual_seed` 复现、连续两次调用不同；`Sampler`/`DraftSampler` 在 CPU 上走 torch 路径、和核的结果一致。往核里注入 19 种变异（下标不加片偏移、噪声不按行偏移、seed 固定、Gumbel 公式少一层 log 或符号反、温度不钳、log-sum-exp 少加 M 或少乘 exp、片数向下取整、T=0 不取 greedy 等）：16 种被测试抓住；剩下 3 种里 1 种是等价变异（被掩码片的 sum 取什么都乘 0），2 种是并列时取右边的下标——解释器用 numpy 的 argmax，不理会 `tie_break_left` 参数，只能在真 GPU 上由 GPU 作业里的同一组测试抓。没有 CUDA 的机器上 `Sampler`/`DraftSampler` 仍然走原来的编译 torch 版本（`torch_forward`），所以 CPU 全量测试不受影响。
- 预期和验证：第六次 A40 作业 `nanovllm-sampled6-a40.sbatch`（集群脚本，不在仓库里） 跑 GPU 测试（这次加上 `tests/test_sampler.py` 和 `tests/test_triton_sampler.py`，在真 GPU 上跑核）、`bench_sampler.py --profile`（HEAD 对 62ddc96 对 a5c238c 同场）和 17 行端到端。要看的数：`DraftSampler` batch 32 从 213 µs 降到多少，chat batch 32 采样对 argmax 每步的差从 +1 ms 变成多少，T=0 对照是否仍然逐 token 相同。

### 复测 5（作业 996212，2026-11-09，代码 ad6d386 = 第一版 Triton 核，51 分钟，300 W 节点）

脚本 `nanovllm-sampled6-a40.sbatch`：GPU 测试这次把 `tests/test_triton_sampler.py` 和 `tests/test_sampler.py` 也放到真 GPU 上跑（84 过、1 败、20 跳，见下），`bench_sampler.py --profile` 跑 HEAD、62ddc96（另加一遍 `TORCHINDUCTOR_COOPERATIVE_REDUCTIONS=1`）和 a5c238c，再跑 17 行。原始行在 `docs/benchmarks/qwen3-8b-a40-sampled.txt` 第五段，微基准摘要在 `results/996212/`。

| 每次调用（batch 32） | Triton 核 ad6d386 | 62ddc96 | 62ddc96 + cooperative | a5c238c |
|---|---|---|---|---|
| `Sampler` 墙钟 / GPU µs | 125.8 / **64.5**（2 个 kernel） | 167.3 / 167.2（5 个） | 166.2 / – | 197.2 / 191.9 |
| `DraftSampler` 墙钟 / GPU µs | 124.1 / **64.4**（2 个） | 209.7 / 207.4（6 个） | 209.2 / – | 219.4 / 214.6 |
| `RejectionSampler` one-hot / 采样，墙钟 µs | 470.9 / 735.6（没改） | 469.7 / 732.5 | 543.2 / 737.2 | 584.2 / 995.9 |

batch 8：两个采样器 GPU 时间 20.4 µs（62ddc96 是 51.0 / 55.7），墙钟 123 µs；batch 1：GPU 6.9 µs（12.5 / 16.8），墙钟 132 / 123 µs。第一级 kernel `_sample_tiles_kernel` batch 1 / 8 / 32 是 4.0 / 17.5 / 60.6 µs，第二级 2.8–4.0 µs。

- **GPU 时间降到了三分之一**：batch 32 上 `DraftSampler` 207 → 64 µs、`Sampler` 167 → 64 µs，launch 数 6 → 2。墙钟只从 210 降到 124：剩下的 ~60 µs 是 Triton 两次启动加 8 个 `torch.empty` 的 CPU 开销，batch 1 上 GPU 只用 7 µs、墙钟还是 120 多。在引擎里这部分被 GPU 的 60 ms 一步盖住（CPU 跑在前面，一步只在末尾同步一次），所以每步真正省的是 GPU 时间：4 × 143 µs ≈ 0.57 ms。
- **端到端对上了**：chat batch 32 采样对 argmax 每步 60.3 对 59.6、60.0 对 59.4 ms（多 0.6–0.7；281207 是 0.6–1.6，279567 是 0.8–1.5），batch 8 44.4 对 43.9，batch 1 38.9 对 38.9，混合 66.8 对 66.2。剩下的 0.6 ms 正好是 `DraftSampler` 4 次 64 µs（0.26 ms）加拒绝采样器采样分支比 one-hot 分支多的 260 µs（728 对 467，其中残差那一个 kernel 330 µs）加 bf16 草稿 logits 的写读。吞吐采样 / argmax：chat batch 32 +4.2% / +4.9%、开关两边都是 1140.1、batch 8 +3.7%（接受率 0.413）、batch 1 +5.0%、混合 +1.1%；T=0 对照 1114.4 对 1107.7 tok/s，token 和步数完全相同。
- **`TORCHINDUCTOR_COOPERATIVE_REDUCTIONS=1` 没用**：两个采样器一个数都没变（166 / 209），one-hot 拒绝采样器反而慢 16%——它改的是单行怎么跨块协作，不碰 looped / persistent 的选择，噪声张量也还在。
- **第一级 kernel 还不到带宽**：batch 32 读 9.7 MB 用 60.6 µs 是 160 GB/s，而且 1 / 8 / 32 行是 4.0 / 17.5 / 60.6 µs，和 program 数成正比，是指令吞吐而不是访存或延迟在卡。看 Triton 的 `random.py`：`tl.rand` 调一次 Philox（10 轮）产出 4 个 32 位数却只留 1 个，一个 1024 宽的片要 1024 次 Philox；Inductor 的 `rand_like` kernel 光生成这 486 万个数就花 29 µs，我们的核里同样的量占了一半。改成 `tl.rand4x`：片的四个四分之一各取同一组 256 个计数器的 u0..u3（计数器 = 行 × 词表 + 片偏移 + j，行、片之间不重叠），Philox 调用数除以 4；CPU 解释器下 13 个测试照过；5 种噪声变异（去掉行偏移 / 片偏移、seed 固定、四份用同一个 u、Gumbel 少一层 log）里"去掉片偏移"起初漏网——各片用同一组噪声时每个元素仍是 Gumbel 分布，只是片间相关，500 行的 softmax 频率检验看不出来；把等 logits 的均匀性检验改成一行三片就抓住了（片间噪声相同则最大值永远落在第一片），5/5。片宽和 warp 数也做成了 `NANOVLLM_SAMPLE_BLOCK` / `NANOVLLM_SAMPLE_WARPS`（默认 1024 / 4），下一次作业顺便量 2048/8、4096/8、512/4、1024/8。
- **一个 GPU 测试失败**：`test_samplers_dispatch_…` 里 Triton 核的 log-norm 和编译 torch 路径的在 rtol 1e-5 / atol 1e-4 下对不上（token 一致，CPU 解释器下两边都过）。两边在 GPU 上用的都是近似算术——Triton 的 `/` 是 `div.full.f32`、`exp` / `log` 是 `ex2` / `lg2.approx`，Inductor 走 libdevice——谁偏得多、偏多少，日志里没有数字。测试改成两边各自对 float64 参考（rtol 1e-5、atol 1e-3），`bench_sampler.py --check` 新增：对 64 和 151936 列、bf16 和 f32、6 个温度，打印两条路径对 float64 的最大绝对 / 相对误差和 T=0 的 token 是否等于 `torch.argmax`（含片内、跨片并列），第七次作业先跑它。q 用的是 `exp(logit/T − log_norm)`，log-norm 偏 1e-4 只让 q 偏 0.01%，对接受率没有影响。

### 复测 6（作业 1275031，2026-11-09，代码 18cdad5 = d6a7d8b 的采样器（rand4x），41 分钟，300 W 节点）

脚本 `nanovllm-sampled7-a40.sbatch`：GPU 测试（84 过、1 败、20 跳），`bench_sampler.py --check`，`--profile` 跑 HEAD 和 ad6d386，五组片宽 / warp 数只计墙钟，再跑 15 行（去掉了开关对和 profiler 对）。原始行在 `docs/benchmarks/qwen3-8b-a40-sampled.txt` 第六段，`--check` 的全部输出在 `results/1275031/check.txt`。

| 每次调用 GPU µs（第一级 + 第二级 kernel） | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| ad6d386（`tl.rand`） | 4.0 + 2.8 | 17.4 + 2.8 | 60.5 + 3.8 |
| d6a7d8b（`tl.rand4x`） | 3.8 + 2.8 | 13.2 + 2.8 | **45.0 + 3.9** |

- **Philox 省四分之三，kernel 快 25%**：batch 32 的第一级 60.5 → 45.0 µs（9.7 MB，215 GB/s），batch 8 17.4 → 13.2；每次调用 GPU 时间 64 → 49 µs。墙钟仍是 120 µs 上下（启动开销），引擎里看不见。
- **片宽 / warp 数没有结论**：五组变体只量了墙钟（没开 `--profile`，是脚本的疏忽），batch 32 上 2048/8、4096/8、512/4、1024/8、2048/4 的 `DraftSampler` 分别是 107.7 / 106.8 / 109.7 / 108.0 / 110.1 µs，默认 1024/4 那一遍是 125.3——但 batch 1（GPU 只有 7 µs）上同样差 15–20 µs，说明这是不同进程之间启动开销的抖动，不是 kernel 的差别。默认保持 1024/4；要比片宽得看 profiler 里的 kernel 时间，留给下一次作业（如果还有必要）。
- **精度查清了**：`--check` 下两条路径对 float64 的 log-norm 误差都在 1e-7 相对以内（64 / 151936 列、bf16 / f32、T = 0.001–2.0 都是），T=0 的 token 两边都等于 `torch.argmax`（含片内、跨片并列和 −inf 行）。上一次作业测试失败的原因是 **Inductor 编译的 torch 路径在 T=0 的行上 log-sum-exp 溢出成 inf**（`logits / 1e-10` 约 1e11，Inductor 的 `logsumexp` 分解在 GPU 上没能稳住，eager 和 CPU 都是有限值），Triton 核在同一行上是有限的、相对误差 5e-8。T=0 的行从不读 log-norm（贪心行的接受判断是 argmax 比较），所以这只影响测试：测试改成 Triton 核对全部行、torch 路径只对 T>0 的行。这次作业的 1 个失败就是改之前的这条断言。
- **端到端**：chat batch 32 每步 argmax 58.8 / 59.2 ms、采样 59.6 / 59.6（多 0.4–0.8；996212 是 0.6–0.7），batch 8 44.0 对 44.3，batch 1 39.0 对 39.0。吞吐采样 / argmax：chat batch 32 +5.4% / +6.1%、batch 8 +0.9%（这次接受率只涨 0.013）、batch 1 +2.9%、混合 +8.5%（接受率 0.569 对 0.548，步数 431 对 488）；T=0 对照 1120.5 对 1120.1 tok/s，token 和步数完全相同。
- **这条线到此为止**：采样草稿的额外开销从 2.8 ms（244661，eager softmax）→ 2.0（278958）→ 0.6–1.6（281207）→ 0.4–0.8 ms（这次），占 60 ms 一步的 1%，而且剩下的能逐项对上账（4 次草稿采样 0.2 ms，拒绝采样器采样分支比 one-hot 多 0.25 ms，草稿 logits 读写）。再往下是把拒绝采样器的采样分支（728 µs，一步一次）也写成 Triton，最多再省 0.3 ms（0.5%），不值一次作业；真要省得把采样器放进 CUDA graph 把启动开销也抹掉，那是另一个工程。

## EAGLE-3 草稿头（2026-11-10 加）

草稿模型那条线的上限是 Qwen3-0.6B 和 8B 有多像（chat 接受率 0.42，每步还要跑 28 层）。EAGLE-3 用一个单层的"头"代替整个草稿模型：头看 target 自己的中间层隐状态来猜下一个 token，训练时就是对着 target 的输出学的，所以 chat 上的接受率高得多（RedHatAI 给的 HumanEval 平均接受长度 k=3 时 2.39、k=7 时 2.69），而且头只有一层。

用的头：`RedHatAI/Qwen3-8B-speculator.eagle3`（Apache-2.0，speculators 格式，vLLM 用 `--speculative-config '{"method": "eagle3"}'` 直接加载的那个）。`config.json` 里要紧的几项：`transformer_layer_config` 是 Llama 风格的单层（hidden 4096、32 头 / 8 KV 头、head_dim 128、intermediate 12288、rms_norm_eps 1e-6、rope_theta 1e6、无 bias），`draft_vocab_size: 32000`（头只在 target 词表里最常用的 32000 个 token 上出 logits），`norm_before_residual: true`，`target_hidden_size: null`（和头一样是 4096）。权重 2.04 GB：`embed_tokens`（151936 × 4096，和 target 的一样）、`fc`（3·4096 → 4096，无 bias，把 target 三层的隐状态拼起来压成一层宽）、`layers.0.*`（`input_layernorm`、`hidden_norm`、`self_attn.{q,k,v,o}_proj`、`post_attention_layernorm`、`mlp.{gate,up,down}_proj`）、`norm`、`lm_head`（32000 × 4096）、两个 buffer `d2t`（草稿词表 → target 词表的偏移，long）和 `t2d`（target 词表里哪些 token 在草稿词表里，bool）。

### 头是怎么算的

- target 的第 2、18、33 层（vLLM 对 36 层的默认 `(2, n//2, n-3)`：这三层的输入，也就是 残差流 `hidden + residual`）拼成 (N, 3·4096)，过 `fc` 得到 (N, 4096) 的"融合隐状态"。
- 头的一行：位置 p 上，输入是 target 在 p 的融合隐状态 和 token `t[p+1]` 的 embedding（`input_layernorm(embeds)` 和 `hidden_norm(hidden)` 各自归一化后拼成 8192 宽进注意力；`norm_before_residual` 为真时残差是归一化后的隐状态），一层 Llama 注意力 + MLP，`norm`，`lm_head` 出 32000 个 logits，预测 `t[p+2]`。也就是说头的行比 target 错一位：target 在位置 p 吃 `t[p]` 预测 `t[p+1]`，头在位置 p 吃 `t[p+1]`（和 target 对 p 的理解）预测 `t[p+2]`。
- 接着往后猜时没有 target 的隐状态了，头就用自己上一行的输出（最后一层的残差流，`norm` 之前的那个）当下一行的隐状态输入，token 用刚猜的草稿；这就是 EAGLE 的"自回归"。
- 注意力：Llama 风格，RoPE 的 theta 按 `transformer_layer_config` 取 1e6（speculators 自带的参考实现里没有传位置编码，vLLM 的 `llama_eagle3` 传了；头是 vLLM 的训练/验证栈出来的，照 vLLM 做）。

### 接进 nano-vllm：隐状态 cache + 头的 KV 复用 target 的块表

草稿模型版（见上）的骨架原样可用：`propose` 先补齐、再逐步猜；头的 KV cache 和 target 用同一张块表、同一组 slot（一层，每块 1 MB）。多出来的是头要 target 的隐状态，而且要的是"上一步算过的那些位置"的隐状态——不是这一步。两个办法：

1. 每步把 target 算出的 (B, k+1, 4096) 融合隐状态留着，下一步按接受数 gather。但不带草稿的步（大 batch 关草稿、自适应 k）、异步调度、抢占后重新 prefill、前缀缓存命中，每一种都要单独处理"隐状态在哪"。
2. **按 slot 存一份隐状态 cache**：target 每算一行，就把这一行的融合隐状态写到 KV 同一个 slot 上（`num_blocks × 256 × 4096` 的 bf16，每块 2 MB，对 8B 每块 37.7 MB 的 KV 是 5%）；头补齐时按补齐位置的 slot 读出来。KV 的记账（哪些位置有效、块什么时候能 hash、抢占后重建）对隐状态同样成立，上面四种情况一个都不用单独处理。写在 target 的 forward 里（`Qwen3Model` 收集三层的残差流后交给头的 `fc`，再按 context 里的 slot_mapping 写），所以在 CUDA graph 里。

选 2。

头的 KV 和草稿模型一样会"掉队"，记账复用 `seq.num_draft_cached_tokens`，但含义改成头的 KV 里有效的位置数 c：
- 头的补齐行是位置 `c .. n-2`（n 是序列现在的 token 数），每行吃 `t[p+1]`，也就是序列最后 `m = n-1-c` 个 token——和草稿模型补齐时的 `tail` 一模一样，只是位置、slot 都往前错一位，再多 gather 一份隐状态；最后一行（位置 n-2，吃最新的 token `t[n-1]`，隐状态是 target 上一步对 n-2 的理解）出第一个草稿。
- 猜第 i 个草稿（i ≥ 2）的行在位置 `n-3+i`，就是 target 给第 i-1 列的位置和 slot；context 长度比 target 的少一。
- 一步带草稿的 decode 之后，c = 这步开始时的 n-1：接受的那几行头虽然在猜草稿时算过（用自己的隐状态），下一步还是用 target 的隐状态重算一遍（vLLM 也这么做：头训练时上下文全是 target 的隐状态，自己猜的那份只是权宜）。所以每步的补齐行数就是上一步接受的 token 数（1 .. k+1），不带草稿的步让它多掉几行，下次一起补。
- 异步调度下 in-flight 的序列：c = in-flight 那步开始时的 n-1，在 GPU 上就是 `num_tokens - num_inflight - 1`。
- prefill：头和 target 同一个 varlen context 跑一遍，id 整体左移一位；一个 chunk 的最后一行吃 prompt 里的下一个 token（还有 chunk 时它是知道的），prompt 已经完整时最后一行没有 token 可吃，填 0，它的 KV 在第一步 decode 的补齐里被重写（c = `num_cached_tokens - 1`）。前缀缓存命中的块里头的 KV 和隐状态都齐（hash 的条件还是 `min(num_cached_tokens, num_draft_cached_tokens)`）。
- 草稿词表：头的 32000 个 logits 按 `d2t` 散到 target 词表宽的 `-inf` 张量里再交给 `DraftSampler` / `RejectionSampler`，两个采样器一行不改；q 在草稿词表之外是 0，`min(1, p/q)` 照样精确。argmax 也在散开后的张量上做（多读 151936 个数，batch 32 下 µs 级）。
- CUDA graph：头的补齐步每序列 1 .. K+1 行（上一步接受了多少），每个行数一组 graph；掉队更多（不带草稿的步之后）就 eager 跑一次。隐状态经 graph 的输入缓冲区进头。
- 没有改的：调度器、块管理器、拒绝采样器、自适应 k、按序列 k（pad 模式）；`speculative_model` 指向头的目录时自动按 `speculators_model_type: "eagle3"` 识别。

### 正确性验证（CPU）

模拟器里加一个假的头（`FakeEagleHead`）代替真头跑真的 `ModelRunner.propose`：
- 假 target 每算一行，把这行的前缀哈希写进假的隐状态 cache（和它写 KV 的同一个 slot）；假头检查补齐行读到的隐状态就是位置 p 的前缀哈希（`tokens[:p+1]`），token 是 `t[p+1]`，位置、slot、context 长度都比草稿模型版少一；没 mask 的行写自己的 KV（另一个种子的哈希），读时逐位置核对；猜草稿的行检查隐状态是自己上一行的输出、token 是上一个草稿。
- 其余检查（输出和参考逐 token 相同、块的 hash 时机、异步、抢占、前缀缓存、自适应 k、pad 模式的 ragged 批）和草稿模型版相同，模式 `eagle1` / `eagle4` / `async_eagle4` / `eagle4_adaptive` / `async_eagle4_adaptive` / `eagle4_acc` / `eagle4_perseq` / `async_eagle4_perseq`，100 个种子每种。
- 模拟器抓到的两处：prefill 完整 prompt 后最后一行的头 KV 还没算（吃的 token 还没采出来），那一块不能在 prefill 这步就 hash（`Scheduler.postprocess` 的 prefill 分支原来按 target 的位置 hash）；异步调度下 in-flight 的步接受了几个 token CPU 不知道，补齐行数只能按它最多能加的算，和草稿模型版固定 2 行是一个道理。
- 前缀缓存共享的块里，最后一个位置的头 KV 是哪条序列先 hash 这块、就用它的下一个 token 算的（块的 hash 只盖到块内的 token）：头的行比 target 错一位，这一位天生不在 hash 里。vLLM 的 EAGLE 也一样（草稿的 KV 用 target 的 slot，前缀命中时不重算）。它只影响草稿质量，不影响正确性；模拟器在块末位置只核对前缀哈希那一半（`FakeModel.head_kv_matches`）。
- 变异体（`eagle` 模式各 12 个种子）：草稿行不错位、草稿步 context 不减一、补齐位置不减一、补齐 context 不减一、把归一化后的输出当隐状态喂回、隐状态从错的 slot 读、补齐少一行、异步 cached 差一、postprocess 沿用草稿模型的规则、prefill 后 c 不退一、prefill 按 target 的位置 hash——11 个全部被抓住。

### 预期

- 头一次 forward ≈ target 的 1/36 层 + 32000 宽的 lm_head + embedding 查表；k=4 时每步 1 次补齐 + 3 次单行，比 0.6B 草稿模型（28 层）便宜得多。
- 接受率：RedHatAI 的数字换算过来，chat 上每步大约接受 1.4–1.7 个草稿（k=3–7），比 0.6B 的 0.42 × k 高；T=0 的输出和不带草稿的 8B 逐 token 相同（这是 GPU 测试的硬检查，头的好坏只影响速度）。

### 实测（作业 1733657，2026-11-10，代码 534ef3f，66 分钟，300 W 节点）

脚本 `cluster/nanovllm-eagle1-a40.sbatch`：先下载头（2.04 GB，16 分钟）、列出 checkpoint 的张量（和上面写的一致：`d2t` 的范围 0–119669、`t2d` 里 32000 个 True），再跑 GPU 测试，然后 25 行吞吐，每种负载 / batch 先跑一行不投机的基线，T=0 的行把输出存下来逐 token 比对。原始行在 `docs/benchmarks/qwen3-8b-a40-eagle3.txt`。单位 tok/s，括号里是对本次基线的倍数和接受率。

| 负载，T=0 | batch | 基线 | 头 k=4 | 头，别的 k | 0.6B 草稿模型 |
|---|---|---|---|---|---|
| chat | 1 | 39.1 | 71.0（×1.82，0.32） | k=6 67.5（×1.73，0.22） | k=4 67.4（×1.72，0.41） |
| | 8 | 293.6 | 431.1（×1.47，0.32） | | k=4 409.5（×1.39，0.41） |
| | 32 | 1046.3 | 1316.6（×1.26，0.33） | k=3 1396.8（×1.33，0.40） | k=4 1115.4（×1.07，0.41） |
| edit | 1 | 38.6 | 105.9（×2.74，0.63） | k=6 110.8（×2.87，0.50） | k=6 124.2（×3.22，0.82） |
| | 32 | 681.6 | 1079.3（×1.58，0.59） | | k=4 997.7（×1.46，0.85） |
| mixed | 32 | 846.2 | 1043.1（×1.23，0.43） | | |

其余的行（都是 chat batch 32，基线 1046.3 / T=0.8 时 1043.8）：异步调度 + 头 k=4 1341.4（×1.28）；K=6 + `min_acceptance=0.3` 1353.4（×1.29，566 步里 392 步 k=2、131 步 k=1）；T=0.8 下按头的分布采草稿 1195.4（×1.15，0.30），argmax 草稿 1220.6（×1.17，0.30）。

- **头在 chat 上赢草稿模型，在 edit 上输。** chat batch 32 头 k=3 ×1.33、k=4 ×1.26，草稿模型 ×1.07；batch 8 ×1.47 对 ×1.39；batch 1 ×1.82 对 ×1.72。edit batch 1 头 k=6 ×2.87，草稿模型 k=6 ×3.22；edit batch 32 头 ×1.58 对 ×1.46。
- **赢在便宜，输在准。** 每步时间 = 行的时间 / 步数。chat batch 32 基线每步 29.1 ms，头 k=4 48.6 ms，草稿模型 60.8 ms：头的 4 次 forward（1 次补齐 + 3 次单行）比 0.6B 的 28 层少 12 ms；batch 1 上头每步比基线多 6.3 ms（31.9 对 25.6），草稿模型多 13.3 ms。但头每步接受的 token 少：chat 上各位的接受率 0.62 / 0.39 / 0.20 / 0.10（平均 1.31 个），草稿模型 0.64 / 0.45 / 0.31 / 0.22（1.62 个）；edit 上 0.80 / 0.66 / 0.52 / 0.40 对 0.92 / 0.86 / 0.83 / 0.80。哪个赢，看每步省下的时间够不够抵接受数的差：chat 上够，edit 上不够。
- **头的接受率为什么掉得快。** 第一位 0.62 和草稿模型持平，往后一位比一位掉得快：第二个草稿起头吃的是自己上一行的隐状态（不是 target 的），误差逐位累积，这是 EAGLE 自回归的本性。edit 上第一位 0.80 也比草稿模型的 0.92 低一截：草稿词表只有 32000 个 token，源文件里的标识符、标点组合、缩进有不少在词表之外，头永远提不出它们（q = 0，必被拒），而 edit 正是复述输入的负载。RedHatAI 公布的 HumanEval 平均接受长度（k=3 时 2.39，含最后那个 bonus token）换算成每步接受的草稿数是 1.39，这里 chat k=3 是 1.19，一个量级。
- **k 怎么选。** chat batch 32 上 k=3 比 k=4 快（1396.8 对 1316.6）：第四位只有 0.10 的接受率，抵不过头多跑一次 forward 加 target 多算一列。自适应开关 K=6 + `min_acceptance=0.3` 自己落到了 k=2 和 k=1，1353.4，比固定 k=3 略低（探测步的代价），说明 batch 32 上 k=2 也该量一下（下个作业）。edit batch 1 上 k=6 比 k=4 快（110.8 对 105.9），和草稿模型一样。
- **采样草稿在头上不赚。** T=0.8 的 chat batch 32，按头的分布采草稿 1195.4，argmax 草稿 1220.6，接受率 0.296 对 0.303。草稿模型那条线上采样赢 4–8%，前提是 q 和 p 接近；头的 q 从第二位起就是自己猜的隐状态算出来的，离 target 的 p 远，从 q 里采等于把 argmax 的把握换成了噪声。这是推测，没有单独验证；默认值不改（对草稿模型是对的），用头时加 `--greedy-drafts`。
- **异步调度照常有效：** ×1.26 → ×1.28（每步 48.6 → 47.8 ms），和草稿模型版的收益一个量级。
- **正确性。** T=0 下头 k=4 和草稿模型 k=4 的输出在 chat batch 1 上 8 条共 3905 个 token 逐个相同；batch 8 有 2/16 条、edit batch 32 有 3/64 条、chat batch 32 有 24/64 条在第 450–490 个 token 之后分叉。和基线比，两种草稿都是 64/64 条在同一个位置（第 0 条的第 75 个 token）分叉，头 k=4 和 k=3 也在那里分叉。分叉跟着 q（一步算几列）和 batch 的组成走，不跟着草稿来源走：bf16 下 (B, k+1) 和 (B, 1) 两种形状的 GEMM 在接近平局的位置各取一边，复测 2 的 T=0 对照组（同 k 时逐 token 相同）已经确认过这一点。GPU 测试的 eagle 模式（头给 8B 打草稿，贪心输出和 transformers 的 bf16 参考逐 token 核对）3 个全过。
- **测试夹具的问题。** GPU 测试在一个进程里按模式依次建引擎，eagle 过了之后 async_eagle 和 eagle_acceptance 在建引擎时断言 `num_kvcache_blocks > 0` 失败（KV 预算 = 显存 × 占比 − 已用 − 本进程的峰值分配），0.6B 上的草稿模型回归也一样（sync 之后的每个模式）。原因是上一个引擎的显存还没还：`LLMEngine.exit` 只 `del` 了 runner，对象图里有环，真正释放要等垃圾回收，而下一个引擎先量了显存。之前的全套 GPU 测试没碰上，是因为模式之间跑了很多测试，回收早就发生了；这次 `-k` 只选了几条，sync 模式的测试被跳过，拆了就建。修法：`ModelRunner.exit` 显式删掉模型、两套 KV cache、隐状态 cache、CUDA graph 和它们的输入缓冲，再 `gc.collect()` + `empty_cache()`；夹具拆完也回收一次；断言带上三个数。下个作业每个 eagle 模式单独一个 pytest 进程，每次之间打印显存占用，两头保险。
- 每行的 recompile 数和草稿模型的行一样（基线 11、k=4 18、开关 23），都在启动预热里；头没有带来新的编译。

### 复测（作业 1734939，2026-11-11，代码 fabf36c，36 分钟，300 W 节点）

脚本 `cluster/nanovllm-eagle2-a40.sbatch`：先验证夹具的修法，再补上一次缺的 k。原始行在 `docs/benchmarks/qwen3-8b-a40-eagle3-k.txt`。

- **夹具修好了。** 三个 eagle 模式各自一个进程（每个 3 过 2 跳），0.6B 上的草稿模型回归照上次那样在一个进程里跑完 15 个模式（15 过 10 跳，上次从第二个模式起全挂）；每次 pytest 之间 `nvidia-smi` 显示显存占用 0 MiB，说明引擎退出时显式释放是够的。
- **头在 chat 上要 k=3（batch 32 和 8 都是），k=2 只差一点。** batch 32（基线 1045.8）：k=2 1396.7（×1.34）、k=3 1418.2（×1.36）、上次的 k=4 1316.6（×1.26）；batch 8（基线 293.3）：k=2 430.9（×1.47）、k=3 444.0（×1.51）、k=4 430.5（×1.47）。每步时间 batch 32：k=2 39.5 ms、k=3 41.2、k=4 48.6（上次）。k=2 → k=3 多一次单行 forward（1.7 ms），换来每步多接受 0.20 个 token；k=3 → k=4 却多了 7.4 ms，远不止一次 forward：batch 32 下 q=4 是 128 行，q=5 是 160 行，target 的 GEMM 从一个 128 行的 tile 变成两个，第四位 0.10 的接受率抵不过这一跳（推测，没有单独量 target 的那一步）。
- **异步调度在头上值 4%：** batch 32 k=2 ×1.34 → ×1.39（1455.0），k=3 ×1.36 → ×1.41（1470.9），每步 41.2 → 39.6 ms。草稿模型版上异步只值 2%：头的每步更短，CPU 那几毫秒占比更大。
- **混合负载 k=3 ×1.35**（1137.1，上次 k=4 ×1.23）：chat 那半要 k=3，edit 那半 k=4 也只多一点，k=3 是 batch 级的折中。
- **edit batch 32 k=6 反而慢**（1004.2，×1.47；k=4 1083.0，×1.59）：第五、六位接受率 0.28 / 0.21，每步多两次 forward 加 target 从 160 行到 224 行，多接受的 0.44 个 token 不够抵。edit batch 1 上 k=6 赢 k=4 是因为 batch 1 下 target 多算几列几乎免费。
- 输出：同 k 的同步 / 异步两次运行在第 450–500 个 token 之后分叉（k=2 8/64 条、k=3 32/64 条），前 450 个 token 每条都相同；异步改变了 batch 的组成，和上一节的解释一致。

结论：头的默认用法是 k=3（chat、混合负载），edit 用 k=4；加异步调度。README 的数字按这次更新。

## 还没做的

- GPU 上 Qwen3-8B 的吞吐对比：已做（见上）。原文：注意力的 `(B, k+1)` 形状、CUDA graph、编译后的拒绝采样器都只能在 GPU 上验证。A40 作业脚本 `cluster/nanovllm-a40.sbatch` 会跑 GPU 测试（贪心结果和 transformers 对比，同步、异步、投机三种模式），以及吞吐对比：随机 prompt（ngram 基本猜不中，看最坏情况）、代码修改任务（让模型给源文件加 docstring 后原样输出）和开放问答，batch 1/8/32，k=0/2/4。草稿模型的 GPU 测试（0.6B 给自己打草稿，接受率应接近 1）在 `tests/test_model_gpu.py` 的 `draft` 模式里。
- ngram 版和异步调度合并：要把 ngram 查找搬到 GPU 上（vLLM 的做法是 CPU 上查、但等上一步落地，等于放弃了异步）。草稿模型版已经合并（见上）。
- k 自适应：按 batch 大小关草稿已做并复测（batch 32 关草稿时 edit ×0.98、chat ×1.00）；按每个草稿位的接受率调 k 已做并同场复测（草稿模型上 edit batch 1 ×3.2、chat batch 32 ×1.19，固定 k=4 只有 ×1.04；n-gram 在 chat batch 32 上从 ×0.74 回到 ×0.99，差距查清是三条 `torch.compile` 守卫，见上）。每条序列一个 k（混合负载需要）已做并在 A40 上量完（见上）：机制对、接受率涨，但哪一格都没赢过 batch 级开关——varlen 丢 CUDA graph，pad 算的行和草稿步数与 batch 用 k_max 一样多，而混合负载里"白算"的位置在读权重为主的 decode 下本来只值两个点；要赢得把 ragged 注意力放进 CUDA graph（按总行数分桶）并让草稿模型只给需要的序列跑需要的步数，还没做。还没做的还有把 batch 大小的阈值按 target 何时算力受限来定（这台 A40 上 8B + 0.6B 用 8 太低），让 K 本身跟着负载走，以及按行数在 varlen 和 pad 之间自动选。
- 从草稿模型的分布采样草稿（而不是 argmax），按 `min(1, p/q)` 接受：已做并在 A40 上量完（见上）：chat 接受率 0.39 → 0.42，吞吐 batch 8 +2–7%、batch 32 +4–8%，edit 不变，T=0 一个 token 都不变；每步多出的 2 ms 查到是 Inductor 不切分带 argmax 的归约，采样器先改成两级分块归约（e88f54d、62ddc96），三次 A40 作业量下来 `Sampler` 和拒绝采样器快了 15–26%，`DraftSampler` 却纹丝不动——Inductor 对 persistent / looped 的选择控制不了、噪声张量去不掉；`DraftSampler` 和 `Sampler` 在 GPU 上改成手写的 Triton 融合核（`fused_sample`，噪声寄存器内生成、logits 只读一遍），第六次 A40 作业（996212）量到两个采样器的 GPU 时间 207 / 167 → 64 µs、每步差距收窄到 0.6 ms；Philox 改成 `rand4x` 后 kernel 再快 25%（作业 1275031：GPU 49 µs 一次），每步差距 0.4–0.8 ms，占一步的 1%，能逐项对账；这条线到此为止（见复测 6）。
- EAGLE-3 草稿头：已做并在 A40 上量完（见上，两次作业）。chat batch 32 ×1.36（k=3；加异步 ×1.41）、batch 8 ×1.51、batch 1 ×1.82，edit batch 32 ×1.59、batch 1 ×2.87，混合 ×1.35；chat 上赢 0.6B 草稿模型（×1.07 / ×1.72），edit 上输给它（×3.22），原因是头便宜但接受率低（自回归的隐状态逐位累积误差、32000 的草稿词表）。还可以做的：头的 `embed_tokens` 和 target 的一样，共享省 1.2 GB；前缀缓存共享块末位那一位的头 KV；让 K 跟着接受率自动落到 3（开关现在落到 2）。
- ngram 索引在 CPU 上：长 prompt 第一次 decode 要建索引，每个 token 4 次字典写入。

## 面试时可以讲的取舍

- 为什么 one-hot 提议下拒绝采样是精确的：接受概率 `min(1, p/q)` 在 q 是 one-hot 时就是 `p(d)`；残差 `max(0, p-q)` 就是把 d 的概率置零。任何只依赖上下文的确定性猜法都满足这个条件。 反过来，有草稿模型时按 q 采样再按 `min(1, p/q)` 验证，接受率从 `p(argmax)` 升到 `Σ min(p, q)`，输出分布不变——"精确"和"接受率"是两回事，one-hot 只是把后者放弃了。A40 上量到的是：接受率涨多少取决于草稿模型的分布和 target 有多像，Qwen3-0.6B 对 8B 在 chat 上只多 0.03，吞吐 +5%，而存 q 的开销不加编译就吃掉一半；0.6B 给自己打草稿时接受率 > 0.9。
- 为什么 batch 里所有序列都是 k+1 个 query：形状统一才能用 `flash_attn_with_kvcache` 和 CUDA graph。代价是没猜中的序列也算 k+1 个位置；vLLM 用 varlen 注意力支持每条序列草稿数不同。后来加的 `speculative_per_sequence` 就是这条路，A40 上量完的结论是在这个组合上不值：读权重为主的 decode 里多几行接近免费，砍掉 chat 一半的位置最多省两个点，而 ragged 步要么丢 CUDA graph（batch 8 上值 25%）要么（补空行）算得和不砍一样多。能说清"为什么没赢、上限是多少、要赢还缺什么"比一个赢了的数字更有用。
- 为什么不回收多分配的块：下一步写的位置从 `L+a` 开始，到 `L+a+k` 结束，需要的块数不会少于这一步，回收了马上又要分配。
- EAGLE 头和草稿模型怎么选：头一次 forward 是一层加 32000 宽的 lm_head，比 0.6B 的 28 层便宜一半以上，但第二个草稿起吃的是自己的隐状态，接受率逐位掉得快，草稿词表之外的 token 提不出来。decode 每步的时间省得多（chat）就头赢，接受率差得多（复述输入的 edit，接受率 0.85 对 0.6）就草稿模型赢；两个都要存一套 KV（头 1 层，模型 28 层）。
- 什么时候有用：decode 受内存带宽限制（小 batch、大模型）而且输出复述输入的时候。随机文本或大 batch 下会变慢，这是 benchmark 里要老实展示的。
