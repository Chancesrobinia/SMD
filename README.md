# SMD-SMAC-Adapter

支持22张标准SMAC地图的SMD (Structured Mask Distillation) 方法训练适配器。

## 🎯 项目状态

✅ **完成** - 所有22张标准SMAC地图已全部支持SMD方法训练

## ✨ 特性

- 🗺️ **完整支持**: 22张标准SMAC地图全部支持
- 🔧 **灵活配置**: 通用脚本 + 地图特定脚本的双层架构
- 📊 **优化参数**: 针对不同地图类型自动调整训练参数
- 🧪 **测试完备**: 包含验证脚本和自动化测试
- 📖 **文档齐全**: 详细的使用指南和API文档

## 🚀 快速开始

### 1. 验证环境

```bash
# 运行验证脚本
python3 verify_smd_support.py
```

预期输出：
```
🎉 恭喜！所有22张标准SMAC地图都已支持SMD方法训练！
```

### 2. 训练示例

```bash
cd on-policy-main-immaenh-serial-graph/onpolicy/scripts/train_smac_scripts

# 训练 3m 地图
./train_smac_3m_SMD.sh

# 使用通用脚本训练 8m 地图
./train_smac_SMD.sh 8m

# 自定义参数训练
export CUDA_VISIBLE_DEVICES=0
export NUM_ENV_STEPS=20000000
export ROLLOUT_THREADS=16
./train_smac_MMM_SMD.sh
```

### 3. 快速测试

```bash
# 运行快速测试（不实际训练）
./test_smd_scripts.sh
```

## 📋 支持的地图

### Marines 系列 (7张)
- `3m`, `8m`, `25m`
- `5m_vs_6m`, `8m_vs_9m`, `10m_vs_11m`, `27m_vs_30m`

### MMM 系列 (2张)
- `MMM`, `MMM2`

### Stalkers and Zealots 系列 (6张)
- `2s3z`, `3s5z`, `3s5z_vs_3s6z`
- `3s_vs_3z`, `3s_vs_4z`, `3s_vs_5z`

### 其他地图 (7张)
- `1c3s5z` - Colossi, Stalkers & Zealots
- `2m_vs_1z` - Marines vs Zealot
- `corridor` - 走廊地图
- `6h_vs_8z` - Hydralisks vs Zealots
- `2s_vs_1sc` - Stalkers vs Spine Crawler
- `so_many_baneling` - Zealots vs Banelings
- `bane_vs_bane` - Banelings vs Banelings
- `2c_vs_64zg` - Colossi vs Zerglings

## 📚 文档

- **[SMD_TRAINING_GUIDE.md](SMD_TRAINING_GUIDE.md)** - 完整训练指南
  - 详细的使用说明
  - 环境变量配置
  - 性能优化建议
  - 故障排除

- **[SUMMARY.md](SUMMARY.md)** - 修改总结
  - 完成的工作
  - 技术实现
  - 项目结构

## 🛠️ 核心组件

### 训练脚本

1. **通用脚本**: `train_smac_SMD.sh`
   - 支持所有22张地图
   - 自动参数优化
   - 灵活的环境变量配置

2. **地图特定脚本**: `train_smac_<map>_SMD.sh`
   - 每张地图一个简洁的包装脚本
   - 调用通用脚本并传递地图名称

### 验证工具

- **verify_smd_support.py** - 自动验证所有地图的SMD支持
- **test_smd_scripts.sh** - 快速测试脚本功能

## 🔑 SMD 关键特性

SMD (Structured Mask Distillation) 方法包含以下核心特性：

1. **异构图神经网络** (`--use_hetero_graph`)
   - 区分友方和敌方单位
   - 使用不同的编码器处理不同实体

2. **门控融合** (`--use_gated_fusion`)
   - 自适应信息融合
   - 提高模型表达能力

3. **结构化掩码蒸馏** (`--use_smd`)
   - 学生网络学习稀疏注意力掩码
   - 教师网络提供密集监督
   - 平衡性能和效率

## 📊 配置参数

### 环境变量

```bash
# 基础配置
CUDA_VISIBLE_DEVICES=0        # GPU 设备
SEED=1                        # 随机种子
NUM_ENV_STEPS=10000000        # 训练步数

# 并行配置
ROLLOUT_THREADS=8             # 环境并行数
TRAINING_THREADS=1            # 训练线程数

# SMD 参数
SMD_CANDIDATE_TOP_K=3         # 候选节点数
SMD_EDGE_TOP_M=1              # 边选择数
SMD_PSEUDO_TOP_M=1            # 伪标签数
LAMBDA_SMD_DIFF=0.01          # 差异损失权重
LAMBDA_SMD_DISTILL=0.05       # 蒸馏损失权重
LAMBDA_SMD_SPARSE=0.001       # 稀疏损失权重
```

### 命令行参数

```bash
./train_smac_SMD.sh <map_name> \
    --num_env_steps 5000000 \
    --ppo_epoch 10 \
    --seed 42 \
    --user_name your_name
```

## 📁 项目结构

```
SMD-SMAC-Adapter/
├── on-policy-main-immaenh-serial-graph/
│   ├── onpolicy/
│   │   ├── scripts/train_smac_scripts/    # 训练脚本目录
│   │   ├── envs/starcraft2/               # SMAC环境
│   │   └── algorithms/r_mappo/            # SMD算法实现
│   └── tests/                             # 测试文件
├── verify_smd_support.py                  # 验证工具
├── test_smd_scripts.sh                    # 测试脚本
├── SMD_TRAINING_GUIDE.md                  # 训练指南
├── SUMMARY.md                             # 修改总结
└── README.md                              # 本文件
```

## 🔍 验证测试

所有地图都通过了以下测试：

- ✅ 脚本存在性检查
- ✅ Shell 语法正确性
- ✅ SMD 必需参数验证
- ✅ 通用脚本配置检查
- ✅ 干运行测试

运行完整测试：

```bash
# 验证所有地图支持
python3 verify_smd_support.py

# 快速功能测试
./test_smd_scripts.sh
```

## 💡 使用示例

### 基础训练

```bash
# 训练 3m 地图
./train_smac_3m_SMD.sh

# 训练 MMM 地图
./train_smac_MMM_SMD.sh
```

### 自定义配置

```bash
# 使用不同的 GPU
export CUDA_VISIBLE_DEVICES=1
./train_smac_8m_SMD.sh

# 增加并行环境数
export ROLLOUT_THREADS=16
./train_smac_25m_SMD.sh

# 快速测试运行（减少训练步数）
export NUM_ENV_STEPS=100000
export USE_EVAL=0
./train_smac_3m_SMD.sh
```

### 高级配置

```bash
# 调整 SMD 参数
export SMD_CANDIDATE_TOP_K=5
export SMD_EDGE_TOP_M=2
export LAMBDA_SMD_DISTILL=0.1
./train_smac_corridor_SMD.sh

# 使用 wandb 记录
export USE_WANDB=1
export EXP_NAME=my_smd_experiment
./train_smac_MMM2_SMD.sh
```

## 🔧 依赖要求

- Python 3.6+
- PyTorch 1.8+
- SMAC 环境
- StarCraft II
- 其他依赖见 `requirements.txt`

## 📖 相关资源

- [SMAC 环境](https://github.com/oxwhirl/smac)
- [MAPPO 论文](https://arxiv.org/abs/2103.01955)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

请参考项目许可证文件。

---

**最后更新**: 2026年9月14日  
**状态**: ✅ 生产就绪，所有22张SMAC地图已完全支持SMD训练
