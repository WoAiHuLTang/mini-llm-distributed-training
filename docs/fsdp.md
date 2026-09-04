# FSDP：Fully Sharded Data Parallel

## 一句话

FSDP 把模型的**参数、梯度、优化器状态**分片（shard）到多张卡上，每卡只存一部分。需要某个参数时，通过 **AllGather** 临时取回完整参数参与计算，用完即释放。这让单卡显存大幅下降，能训练单卡放不下的模型。

## 与 DDP 的本质区别

| | DDP | FSDP |
| --- | --- | --- |
| 参数存储 | 每卡完整副本 | 分片到各卡 |
| 梯度 | 每卡完整，AllReduce 求平均 | 分片，ReduceScatter 归约 |
| 优化器状态 | 每卡完整 | 分片 |
| 单卡显存 | 高（≈单卡） | 低（随卡数下降） |
| 通信 | AllReduce（每步一次） | AllGather + ReduceScatter（每层多次） |
| 适用 | 模型能放进单卡 | 模型放不进单卡 / 显存受限 |

## 前向：AllGather

FSDP 按"分片单元"（通常是一个 Transformer Block）组织。前向进入某层前，需要该层的**完整参数**：

```
参数分片（本地只存 1/N）
      │
      ▼
AllGather（从所有 rank 收集，拼成完整参数）
      │
      ▼
完整参数参与该层计算
      │
      ▼
计算完立即释放完整参数（只保留分片）
```

## 反向：ReduceScatter

反向时，各卡算出该层**完整梯度**，但只需要自己那份分片，于是用 **ReduceScatter** 归约并只保留本地分片：

```
该层完整梯度
      │
      ▼
ReduceScatter（跨卡求和，每卡只留自己的 1/N 分片）
      │
      ▼
本地梯度分片
```

## 为什么 FSDP 显存低但吞吐通常低于 DDP？

- **显存低**：参数 + 梯度 + 优化器状态都被分片，单卡只存 1/N。
- **吞吐低**：每层前向/反向都要 AllGather / ReduceScatter，通信**次数多**（每层一次），且更难与计算重叠；相比 DDP 每步只有一次 AllReduce，通信开销更大。

## Activation Checkpointing（激活检查点）

即使参数分片了，**中间激活**（activation）仍可能占大量显存。Activation Checkpointing 的做法是：

- 前向时**不保存**中间激活，只保存少量 checkpoint；
- 反向需要梯度时**重新前向计算**这些激活。

```
前向：不存激活 ──► 只存 checkpoint（省显存）
反向：需要激活 ──► 重算前向（多花计算）──► 算梯度
```

**代价**：训练变慢（约多 30% 计算），**收益**：显存进一步大幅下降。
FSDP + AC 是"显存最低、训练最慢"的配置，适合超大模型。

## 什么时候用 FSDP？

- 模型**放不进单卡**，或想用更小 batch 塞进更多内容。
- 显存是首要瓶颈，愿意用吞吐/通信换显存。
- 需要训练超大 LLM 的常见选择（配合 AC）。

## 本项目实现

见 [`src/mini_llm/distributed/fsdp.py`](../src/mini_llm/distributed/fsdp.py)：

- 用 `transformer_auto_wrap_policy` 在 **TransformerBlock** 边界自动分片（LLM 标准做法）。
- 支持 `full_shard`（ZeRO-3 风格）与 `shard_grad_op`（ZeRO-2 风格）。
- 通过 `apply_activation_checkpointing` 对每个 TransformerBlock 施加非重入式激活检查点。
- 内置 BF16 / FP16 混合精度策略。
