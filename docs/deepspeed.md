# DeepSpeed：ZeRO-2 / ZeRO-3

## 一句话

DeepSpeed 是微软开源的分布式训练框架，通过 **ZeRO（Zero Redundancy Optimizer）** 把训练状态（优化器状态、梯度、参数）分片到多卡，从而大幅降低单卡显存。与 DDP/FSDP 不同，DeepSpeed 用一个 **engine** 接管整个训练步（forward / backward / optimizer step），我们只需调用 `engine.backward(loss)` 和 `engine.step()`。

## DeepSpeed engine 与自建优化器的区别

在 Single / DDP / FSDP 中，我们**自己**创建优化器并手动调用 `loss.backward()` → `optimizer.step()`。DeepSpeed 则把这一切封装进 engine：

```python
engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_cfg, ...)
loss = engine(input_ids, targets=target_ids)   # forward（engine 就是被包装的模型）
engine.backward(loss)                          # backward + 梯度归约
engine.step()                                  # 梯度裁剪 + 优化器更新
```

因此在本项目中，`deepspeed` 策略下 `self.model` 是 engine，`self.optimizer` 由 `wrap_deepspeed` 返回，训练步走 engine 分支，且**外层 autocast 被禁用**（DeepSpeed 通过 JSON 配置内部管理 bf16/fp16 精度）。

## ZeRO 的三个 stage

| | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- |
| 优化器状态 | 分片 | 分片 | 分片 |
| 梯度 | 每卡完整 | 分片 | 分片 |
| 模型参数 | 每卡完整 | 每卡完整 | **分片** |
| 单卡显存 | 中 | 低 | 最低 |
| 通信 | AllReduce | ReduceScatter（梯度） | AllGather（参数）+ ReduceScatter（梯度） |

- **ZeRO-2**：只分片优化器状态 + 梯度，**参数仍是每卡全量副本**。前向无需 AllGather 参数，通信主要是反向时的 ReduceScatter，比 FSDP 更轻。
- **ZeRO-3**：连参数也分片，每次 forward/backward 都要 **AllGather 参数 + ReduceScatter 梯度**，通信量最大，但能装下单卡完全放不下的超大模型。

## 本项目如何接入

1. **JSON 配置**（`configs/deepspeed_z2.json` / `deepspeed_z3.json`）用 `zero_optimization.stage` 指定 2 或 3。
2. **封装**（`src/mini_llm/distributed/deepspeed.py`）读取 JSON、补全 batch size / 优化器 / 精度，再调用 `deepspeed.initialize` 返回 engine。
3. **Trainer**（`src/mini_llm/trainer.py`）把 `deepspeed` 当作一种策略：engine 接管 backward/step，跳过自建优化器与 autocast。
4. **入口**：`benchmarks/benchmark.py` 与 `src/mini_llm/train.py` 增加 `--strategy deepspeed --ds-config <json>`。

```bash
# 训练（ZeRO-2）
bash scripts/train_deepspeed.sh 2 8 3 2
# 训练（ZeRO-3）
bash scripts/train_deepspeed.sh 2 8 3 3

# benchmark（ZeRO-2 / ZeRO-3）
torchrun --nproc_per_node=2 benchmarks/benchmark.py \
    --strategy deepspeed --config configs/gpt_small.yaml \
    --ds-config configs/deepspeed_z2.json
torchrun --nproc_per_node=2 benchmarks/benchmark.py \
    --strategy deepspeed --config configs/gpt_small.yaml \
    --ds-config configs/deepspeed_z3.json
```

## 实测对比（2 × RTX 4090, bf16, micro_batch=8, ~110M 参数）

| Strategy | GPUs | Peak Memory/GPU (GB) | tokens/s | step_ms | Scaling Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDP | 2 | 6.37 | 39,785 | 103.0 | 37.8% |
| FSDP | 2 | 5.15 | 23,378 | 175.2 | 22.2% |
| DeepSpeed ZeRO-2 | 2 | 5.43 | 45,052 | 90.9 | 42.8% |
| DeepSpeed ZeRO-3 | 2 | 4.71 | 24,980 | 164.0 | 23.7% |

**结论**：

- **ZeRO-2 是 2 卡分片策略中吞吐最高的**（45,052 tokens/s，扩展效率 42.8%）：参数全量副本使前向无需 AllGather，通信比 FSDP 轻，显存 5.43 GB 略高于 FSDP 但吞吐明显更高。
- **ZeRO-3 显存最低（4.71 GB）但吞吐骤降**：参数分片导致每次 forward/backward 都要 AllGather + ReduceScatter，通信量最大，吞吐与 FSDP 相当（24,980 tokens/s）。
- 本模型很小（单卡 ~6GB），分片策略的显存优势有限，反而被通信开销拖累吞吐。ZeRO-3 的真正价值在于**能训练单卡放不下的超大模型**。

## 常见坑

- **版本兼容**：DeepSpeed 0.19.6 在 Python 3.12 下因 Dynamo 不支持而导入失败，需降级到 **0.14.4**。
- **batch size 需显式给定**：本项目不向 `deepspeed.initialize` 传 dataloader，因此 JSON 里的 `"auto"` 无法被解析，需在封装里按 `micro_batch × grad_accum × world_size` 显式算出 `train_batch_size`。
- **engine 接管 step**：不要对 engine 再手动 `loss.backward()` + `optimizer.step()`，否则会重复归约/更新。
