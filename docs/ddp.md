# DDP：Distributed Data Parallel

## 一句话

DDP 让**每个 GPU 都持有一份完整的模型副本**，各自用不同数据做前向/反向，然后在反向过程中通过 **AllReduce** 把梯度在所有卡之间求平均，保证所有副本的模型参数保持一致。

## 训练流程

```
每个 rank（进程/GPU）：
  1. 持有完整模型副本
  2. 用 DistributedSampler 分到的不同 batch 做 forward
  3. backward 计算本地梯度
  4. 梯度按"桶(bucket)"打包，触发 NCCL AllReduce（各卡梯度求和取平均）
  5. 每卡用平均后的梯度更新自己那份参数（所有卡更新一致）
```

## 关键点：梯度 AllReduce

- 反向时，PyTorch 把参数按大小分成若干**梯度桶（bucket）**。
- 当一个桶的所有梯度就绪，立即对该桶发起 **AllReduce**，而不是等全部梯度算完。
- 这种"边算边同步"（bucket + overlap）让通信与计算重叠，显著降低同步等待。

```
Backward 计算梯度
      │
      ▼
梯度桶就绪 ──► NCCL AllReduce（跨卡求和取平均）
      │
      ▼
Optimizer step（每卡用一致梯度更新）
```

## 为什么 DDP 显存 ≈ 单卡？

DDP **不切分模型**，每卡都要放完整参数 + 梯度 + 优化器状态，所以单卡显存几乎不随卡数下降。它省的是**时间**（吞吐），不是**显存**。

## 为什么多 GPU 不一定线性加速？

理想情况下 2 卡吞吐是单卡 2 倍，但实际扩展效率 < 100%，原因：

1. **通信开销**：每步都要 AllReduce，通信时间无法完全被计算掩盖。
2. **负载不均 / 同步等待**：最慢的 rank 决定整步耗时（木桶效应）。
3. **kernel 启动与调度开销**：多进程本身有额外开销。
4. **小模型 / 小 batch**：通信占比相对更高，扩展效率更差。

## 什么时候用 DDP？

- 模型能**完整放进单卡显存**。
- 想要**最高吞吐**、实现简单、通信开销小。
- 显存不是瓶颈，吞吐才是瓶颈。

## 本项目实现

见 [`src/mini_llm/distributed/ddp.py`](../src/mini_llm/distributed/ddp.py)，
通过 `torch.nn.parallel.DistributedDataParallel` 封装，并开启
`gradient_as_bucket_view`（省显存）与可配置的 `bucket_cap_mb`。
