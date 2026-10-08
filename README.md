# Microduck 篮球任务:蒙眼站球策略的后训练改进

基于 [Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground)
(上游 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl))
的盲眼篮球平衡任务,对这个策略做的一轮**强化学习后训练**改进实验。

**任务**:机器鸭站在一颗自由滚动的篮球顶上,策略是蒙眼的(61 维输入里
没有任何球状态),要保平衡并跟踪速度指令。起点为 b11 发布版
(60 秒存活率 97.01%,转向误差 1.26 rad/s)。

## 修改思路

### 先诊断,再动手

分析 b11 的 92 次失败记录:掉落与指令内容无关、随时间均匀发生;
对照实验显示无推扰存活 99.3%、有推扰 87.5%——**失败几乎全部来自推扰**。
又用一个只训预测头的"热身期"实测:LSTM 隐状态对球状态的预测误差
≈ 直接猜均值——**基线记忆里几乎不含球的信息**(这是转向不听话的病根)。

### 三个阶段,两种药方

| 阶段 | 做法 | 属于 | 结果 |
|---|---|---|---|
| **0 推扰对齐** | 训练推扰间隔 1.5–3 s → 0.5–1.5 s,对齐考试强度 | 改数据分布 | 存活 97.01% → **98.99%** |
| **0.5 指令对齐** | 训练指令范围 ×1.5 / ×2,对齐考试指令 | 改数据分布 | 转向仅 1.226 → 1.193,**排除"没练过大指令"假设** |
| **1 猜球头 + KL 锚** | 训练时逼 LSTM 隐状态预测球状态(标签来自训练时特权可见的 critic 观测),同时用 KL 惩罚把行为锁在源策略附近 | 改优化目标 | 存活 **99.48%**、转向 **1.134**,全面超越基线 |

### 关键教训:改目标必须配锚

猜球头的前两版(无锚)直接失败:辅助损失权重 0.5 → 存活崩到 22%;
降到 0.05 并加热身期 → 仍只有 85.6%。原因:辅助梯度与策略梯度抢同一组
LSTM 参数,把已精细调优的平衡策略推离了最优。

第三版加上主流后训练的标准配方——**冻结一份参考策略,每步惩罚
KL(参考 ‖ 当前)**——同样的猜球压力就变成安全的:存活率不降反升,
转向、动作平稳度同时改善。一句话总结:

> **改"练什么"(数据分布)不需要锚;改"优化什么"(目标函数)必须配锚。**

### 最终成绩(3 种子 × 1024 环境 × 60 秒,首次摔倒即失败)

| 策略 | 存活率 | 转向误差 rad/s | 动作 RMS |
|---|---:|---:|---:|
| b11 发布版 | 97.01% | 1.262 | 0.234 |
| 阶段 0(推扰对齐) | 98.99% | 1.226 | 0.234 |
| 猜球头无锚 v2 | 85.6% | 1.177 | 0.256 |
| **猜球头 + KL 锚(本工作)** | **99.48%** | **1.134** | **0.226** |
| 参考:能看球的非盲策略 | 99.12% | 0.757 | 0.137 |

新增代码:`src/mjlab_microduck/basketball_belief.py`(算法:
`BasketballBeliefPPO` = PPO + 猜球辅助头 + KL 锚)、
`scripts/finetune_basketball_belief.py`(续训入口)、
`src/mjlab_microduck/video_effects.py`(从上游 desk-climb 快照恢复的渲染依赖)。

## Quick Start

需要 CUDA GPU 和 [uv](https://docs.astral.sh/uv/)。无头服务器建议全程带
`WANDB_MODE=offline`(日志只写本地)。

```bash
git clone https://github.com/likaizhen-code/microduck-playground
cd microduck-playground
uv sync

# 下载 b11 检查点(Hugging Face)
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('HannesVonEssen/microduck-basketball', local_dir='artifacts/basketball')
PY

# CPU 配置测试(不需要 GPU)
uv run --with pytest pytest tests/
```

### 第一步永远是小试跑(64 环境 × 5 迭代,几分钟)

```bash
WANDB_MODE=offline MICRODUCK_BB_BLIND=1 MICRODUCK_BB_HISTORY=1 \
uv run python scripts/finetune_basketball.py artifacts/basketball/checkpoint.pt \
  --run-name smoke --num-envs 64 --iterations 5 \
  --learning-rate 2e-5 --action-rate-weight -0.2 \
  --push-interval-s 0.5 1.5 --save-interval 5
```

### 复现阶段 0(推扰加密续训,4096 环境 × 500 迭代 ≈ 12 分钟/RTX4090)

```bash
WANDB_MODE=offline MICRODUCK_BB_BLIND=1 MICRODUCK_BB_HISTORY=1 \
uv run python scripts/finetune_basketball.py artifacts/basketball/checkpoint.pt \
  --run-name stage0-push-dense --num-envs 4096 --iterations 500 \
  --learning-rate 2e-5 --action-rate-weight -0.2 --command-scale 1 \
  --episode-seconds 10 --seed 42 \
  --push-interval-s 0.5 1.5 --save-interval 125
```

### 复现阶段 1(猜球头 + KL 锚,从阶段 0 最佳档出发)

```bash
WANDB_MODE=offline MICRODUCK_BB_BLIND=1 MICRODUCK_BB_HISTORY=1 \
uv run python scripts/finetune_basketball_belief.py \
  logs/rsl_rl/basketball/<stage0运行目录>/model_XXXX.pt \
  --run-name stage1-belief-anchor --num-envs 4096 --iterations 500 \
  --learning-rate 2e-5 --action-rate-weight -0.2 --command-scale 1.5 \
  --push-interval-s 0.5 1.5 --save-interval 125 \
  --belief-weight 0.05 --belief-warmup 100 --anchor-weight 1.0
# 训练日志逐轮打印 BELIEF_LOSS(猜球误差)与 ANCHOR_KL(离源策略的漂移)
```

### 导出、对齐、评估、看视频

```bash
RUN=logs/rsl_rl/basketball/<运行目录>

# 导出 ONNX(归一化器烘焙在内,必须走此路径)
MICRODUCK_BB_BLIND=1 MICRODUCK_BB_HISTORY=1 \
uv run python scripts/export.py Mjlab-Basketball-MicroDuck \
  --checkpoint-file $RUN/model_XXXX.pt --onnx-file out.onnx

# PyTorch/ONNX 数值对齐(误差应 ~1e-6)
uv run python scripts/verify_basketball_onnx_parity.py $RUN/model_XXXX.pt out.onnx

# 三种子考试(固定协议:60 秒、首次摔倒即失败)
for SEED in 101 202 303; do
  uv run python scripts/eval_basketball_long.py $RUN/model_XXXX.pt \
    eval_seed${SEED}.json --seed $SEED --seconds 60 \
    --num-envs 1024 --command-scale 2 --blind
done

# 渲染 30 秒验收视频(无头服务器:MUJOCO_GL=egl)
MUJOCO_GL=egl uv run python scripts/render_checkpoint.py \
  --task Mjlab-Basketball-MicroDuck \
  --checkpoint-file $RUN/model_XXXX.pt \
  --out-dir video_check --duration-s 30 --seed 0 --follow-entity ball
```

## 边界声明

- 所有数字均为**仿真内**成绩;未做真机测试(上游对该策略同样如此定位)。
- 评估协议沿用上游发布版定义(3 种子 × 1024 × 60 s,推扰重置基座水平速度
  ±0.09 m/s),保证与 b11 基线可比。
- 许可证与上游一致(Apache-2.0 / 硬件 CC BY-NC-SA 4.0),详见 `LICENSE`、`NOTICE`。
