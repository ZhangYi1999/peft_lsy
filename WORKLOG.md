# CLARE Development Worklog

> 记录每个 session 完成的工作，供下一个 session 参考。

---

## Session: 2026-03-07

### 已完成：Eval 加速 — Group-by-Adapter Batching

**文件：** `peft_lsy/src/peft/tuners/clare/layer.py`

**问题：** `CLARELayer.forward()` 的 eval 分支对 batch 中每个样本单独 forward 一次（B 次循环），即使多个样本路由到同一个 adapter。对 `LoRAMultiheadAttention` 尤其痛：每次 forward 都触发 `merge() → forward() → unmerge()`。

**解决方案：** 将 per-sample 循环替换为 group-by-adapter 批量 forward：
1. 按 `adapter_id` 将样本索引分组 → `dict[adapter_id → list[sample_indices]]`
2. 对每组做一次批量 forward（而非 B 次 1-sample forward）

**改动位置：** `layer.py` eval 分支（原 lines ~303–323 的 for 循环），替换为：

```python
# Group sample indices by routed adapter to enable batched forwarding
adapter_groups: dict[int, list[int]] = {}
for idx, top_1_idx in enumerate(top_1_idx_list):
    adapter_id = self.clare_discriminators[self.adapter_name][top_1_idx].connected_adapter_indices.item()
    if adapter_id not in adapter_groups:
        adapter_groups[adapter_id] = []
    adapter_groups[adapter_id].append(idx)

for adapter_id, sample_indices in adapter_groups.items():
    batch_input = adapter_input[sample_indices]
    if self.use_lora:
        self._activate_lora_adapter(adapter_id)
        if adapter_input.ndim == 3 and not self.module_config.batch_first:
            batch_output = self.base_layer(batch_input.transpose(0, 1), **kwargs).transpose(0, 1)
        else:
            batch_output = self.base_layer(batch_input, **kwargs)
        adapter_result[sample_indices] = batch_output
    else:
        batch_output = self.clare_func_adapters[self.adapter_name][adapter_id](batch_input)
        adapter_result[sample_indices] = batch_output
```

**加速效果：**
- B=1（online eval）：无变化
- B=32，单任务（discriminator 训练好时所有样本路由同一 adapter）：32 次 forward → 1 次 forward
- B=32，4 任务均匀分布：~4 次 batched forward（替代 32 次）
- LoRA merge/unmerge：从 O(B) 降到 O(N_unique_adapters)

---

### 待实现：CLARE Router 可视化（下一 session 继续）

**目标：** 在 online robot eval 中实时可视化每个 CLARELayer 的 router 选择，使用 Rerun 播放视频和 router 决策。

**方案：** 创建独立文件 `lerobot/src/lerobot/scripts/lerobot_eval_clare.py`，在 `rollout()` 的每个 step 中用 Rerun 记录：
- 渲染帧（作为视频流）
- 每层 adapter 选择（scalar 时间序列，按层分轨道）

**关键信息链路：**
```python
# policy.select_action(obs) 调用后：
peft_modules = policy.base_model.adapter_layers   # List[CLARELayer]
for module in peft_modules:
    info = module.info_dicts
    disc_id    = info["top_1_idx_list"][0]         # env 0 (online eval B=1)
    disc       = module.clare_discriminators[module.adapter_name][disc_id]
    task_id    = disc.connected_adapter_task_id.item()
    adapter_id = disc.connected_adapter_indices.item()
    layer_key  = f"{module.layer_name}.{module.layer_id}"
```

**Rerun 日志设计（待确认 Rerun 版本 API）：**
```python
import rerun as rr
rr.init("clare_eval", spawn=True)

# 每 step：
rr.set_time_sequence("step", step)
rr.log("camera/rgb", rr.Image(frame_hwc))
for layer_key, info in routing_step.items():
    rr.log(f"routing/{layer_key}/task_id",    rr.Scalar(info["task_id"]))
    rr.log(f"routing/{layer_key}/adapter_id", rr.Scalar(info["adapter_id"]))
```

**下一步行动：**
1. 检查 Rerun 是否在 lerobot 环境中安装（`pip show rerun-sdk`）
2. 确认 Rerun API 版本（0.x vs 0.16+，API 有差异）
3. 创建 `lerobot_eval_clare.py`：基于 `lerobot_eval.py` 的 `rollout()` 和 `eval_policy()`，注入 Rerun 日志
4. 验证：`python lerobot_eval_clare.py` 跑一个 episode，Rerun viewer 中同时看到视频和 router 轨迹

---

## 背景：CLARE 架构速查

- **`CLARELayer`** (`peft_lsy/src/peft/tuners/clare/layer.py`)：核心层，eval forward 中通过 discriminator 选 adapter
- **`CLAREModel`** (`peft_lsy/src/peft/tuners/clare/model.py`)：`adapter_layers` 属性 → 所有 CLARELayer 的列表
- **`_info_dicts`**：每次 forward 后存 `top_1_idx_list`（路由结果）、`losses`（各 discriminator loss）
- **Discriminator → Adapter 映射**：`discriminator.connected_adapter_indices`（adapter id），`discriminator.connected_adapter_task_id`（task id）
- **训练脚本**：`lerobot/src/lerobot/scripts/clare/clare.py`
