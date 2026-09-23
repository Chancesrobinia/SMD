# SMD-SMAC 训练指南

## 概述

本项目已经为 **22 张标准 SMAC 地图**全部配置了 SMD (Structured Mask Distillation) 方法训练支持。

## 支持的地图列表

### Marines 系列
- `3m` - 3 Marines vs 3 Marines
- `8m` - 8 Marines vs 8 Marines  
- `25m` - 25 Marines vs 25 Marines
- `5m_vs_6m` - 5 Marines vs 6 Marines
- `8m_vs_9m` - 8 Marines vs 9 Marines
- `10m_vs_11m` - 10 Marines vs 11 Marines
- `27m_vs_30m` - 27 Marines vs 30 Marines

### MMM 系列
- `MMM` - Marines, Marauders, Medivacs
- `MMM2` - MMM 困难版本

### Stalkers and Zealots 系列
- `2s3z` - 2 Stalkers & 3 Zealots
- `3s5z` - 3 Stalkers & 5 Zealots
- `3s5z_vs_3s6z` - 3 Stalkers & 5 Zealots vs 3 Stalkers & 6 Zealots

### Stalkers 系列
- `3s_vs_3z` - 3 Stalkers vs 3 Zealots
- `3s_vs_4z` - 3 Stalkers vs 4 Zealots
- `3s_vs_5z` - 3 Stalkers vs 5 Zealots

### 混合单位
- `1c3s5z` - 1 Colossus, 3 Stalkers & 5 Zealots
- `2m_vs_1z` - 2 Marines vs 1 Zealot

### 困难地图
- `corridor` - 6 Zealots vs 24 Zerglings (走廊)
- `6h_vs_8z` - 6 Hydralisks vs 8 Zealots
- `2s_vs_1sc` - 2 Stalkers vs 1 Spine Crawler

### 特殊地图
- `so_many_baneling` - 7 Zealots vs 32 Banelings
- `bane_vs_bane` - 24 Banelings vs 24 Banelings
- `2c_vs_64zg` - 2 Colossi vs 64 Zerglings

## 快速开始

### 1. 使用地图特定脚本

每张地图都有对应的训练脚本：

```bash
cd on-policy-main-immaenh-serial-graph/onpolicy/scripts/train_smac_scripts

# 训练 3m 地图
./train_smac_3m_SMD.sh

# 训练 MMM 地图
./train_smac_MMM_SMD.sh

# 训练 corridor 地图
./train_smac_corridor_SMD.sh
```

### 2. 使用通用脚本

也可以使用通用脚本并指定地图名称：

```bash
./train_smac_SMD.sh <map_name>

# 例如：
./train_smac_SMD.sh 8m
./train_smac_SMD.sh 3s5z
./train_smac_SMD.sh corridor
```

## 环境变量配置

可以通过环境变量自定义训练参数：

```bash
# 基本配置
export CUDA_VISIBLE_DEVICES=0          # 使用的 GPU
export SEED=1                          # 随机种子
export NUM_ENV_STEPS=10000000          # 训练步数

# 并行配置
export ROLLOUT_THREADS=8               # 环境并行数
export TRAINING_THREADS=1              # 训练线程数

# PPO 参数
export PPO_EPOCH=15                    # PPO epoch 数
export NUM_MINI_BATCH=1                # mini-batch 数量
export CLIP_PARAM=0.2                  # PPO clip 参数

# SMD 参数
export SMD_CANDIDATE_TOP_K=3           # 候选节点数
export SMD_EDGE_TOP_M=1                # 边选择数
export SMD_PSEUDO_TOP_M=1              # 伪标签数
export LAMBDA_SMD_DIFF=0.01            # 差异损失权重
export LAMBDA_SMD_DISTILL=0.05         # 蒸馏损失权重
export LAMBDA_SMD_SPARSE=0.001         # 稀疏损失权重
export SMD_TARGET_DEGREE=1.0           # 目标度数

# 评估配置
export USE_EVAL=1                      # 是否评估
export EVAL_EPISODES=32                # 评估回合数

# 日志配置
export USE_WANDB=0                     # 1: 使用 wandb, 0: 使用 TensorBoard
export EXP_NAME=my_experiment          # 实验名称
```

### 使用示例

```bash
# 在 GPU 1 上训练 8m 地图，使用 16 个并行环境
export CUDA_VISIBLE_DEVICES=1
export ROLLOUT_THREADS=16
./train_smac_8m_SMD.sh

# 使用自定义参数训练 MMM 地图
export SMD_CANDIDATE_TOP_K=5
export PPO_EPOCH=10
export NUM_ENV_STEPS=20000000
./train_smac_MMM_SMD.sh

# 快速测试运行（减少训练步数）
export NUM_ENV_STEPS=100000
export USE_EVAL=0
./train_smac_3m_SMD.sh
```

## 命令行参数

也可以直接传递额外的命令行参数：

```bash
./train_smac_SMD.sh 8m \
    --num_env_steps 5000000 \
    --ppo_epoch 10 \
    --seed 42 \
    --user_name your_name
```

## SMD 方法特点

SMD (Structured Mask Distillation) 方法的关键特性：

1. **异构图神经网络** (`--use_hetero_graph`)
   - 区分友方单位和敌方单位
   - 使用不同的编码器处理不同类型的实体

2. **门控融合** (`--use_gated_fusion`)
   - 自适应融合不同来源的信息
   - 提高模型表达能力

3. **结构化掩码蒸馏** (`--use_smd`)
   - 学生网络学习稀疏的注意力掩码
   - 教师网络提供密集的注意力监督
   - 平衡性能和计算效率

## 验证配置

运行验证脚本检查所有地图是否正确配置：

```bash
python3 verify_smd_support.py
```

预期输出：
```
🎉 恭喜！所有22张标准SMAC地图都已支持SMD方法训练！
```

## 目录结构

```
SMD-SMAC-Adapter/
├── on-policy-main-immaenh-serial-graph/
│   ├── onpolicy/
│   │   ├── scripts/
│   │   │   └── train_smac_scripts/
│   │   │       ├── train_smac_SMD.sh           # 通用 SMD 训练脚本
│   │   │       ├── train_smac_3m_SMD.sh        # 3m 地图脚本
│   │   │       ├── train_smac_8m_SMD.sh        # 8m 地图脚本
│   │   │       ├── train_smac_MMM_SMD.sh       # MMM 地图脚本
│   │   │       └── ...                         # 其他地图脚本
│   │   ├── envs/
│   │   │   └── starcraft2/
│   │   │       ├── smac_maps.py                # SMAC 地图配置
│   │   │       └── StarCraft2_Env.py           # SMAC 环境
│   │   └── algorithms/
│   │       └── r_mappo/
│   │           └── algorithm/
│   │               └── r_actor_critic.py       # SMD 算法实现
│   └── tests/
│       └── test_smac_smd_scripts.py            # SMD 脚本测试
└── verify_smd_support.py                       # 验证脚本
```

## 故障排除

### 问题：脚本没有执行权限

```bash
chmod +x on-policy-main-immaenh-serial-graph/onpolicy/scripts/train_smac_scripts/*.sh
```

### 问题：找不到 SMAC 环境

确保已安装 SMAC：
```bash
pip install git+https://github.com/oxwhirl/smac.git
```

### 问题：StarCraft II 未安装

参考 SMAC 文档安装 StarCraft II：
https://github.com/oxwhirl/smac#installing-starcraft-ii

### 问题：内存不足

减少并行环境数：
```bash
export ROLLOUT_THREADS=4  # 默认是 8
./train_smac_SMD.sh <map_name>
```

## 性能建议

1. **小地图** (3m, 8m, 2s3z 等)
   - `ROLLOUT_THREADS=8-16`
   - `NUM_ENV_STEPS=10000000`

2. **中等地图** (MMM, 3s5z, 10m_vs_11m 等)
   - `ROLLOUT_THREADS=8`
   - `NUM_ENV_STEPS=10000000`
   - `PPO_EPOCH=10-15`

3. **大地图** (25m, 27m_vs_30m 等)
   - `ROLLOUT_THREADS=4-8`
   - `NUM_ENV_STEPS=20000000`
   - `PPO_EPOCH=10`

4. **超困难地图** (corridor, 2c_vs_64zg 等)
   - `ROLLOUT_THREADS=8`
   - `NUM_ENV_STEPS=10000000`
   - 可能需要调整算法为 `ALGORITHM_NAME=mappo`

## 参考

- SMAC: https://github.com/oxwhirl/smac
- MAPPO: https://arxiv.org/abs/2103.01955
- 异构图神经网络: 需要补充论文链接
- SMD 方法: 需要补充论文链接
