#!/usr/bin/env bash
# 快速测试 SMD 训练脚本的示例

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "SMD-SMAC 训练脚本快速测试"
echo "======================================================================"
echo

# 测试几个代表性地图的脚本
TEST_MAPS=("3m" "8m" "MMM" "2s3z" "corridor")

echo "测试 ${#TEST_MAPS[@]} 张代表性地图的训练脚本..."
echo

for map in "${TEST_MAPS[@]}"; do
    script="on-policy-main-immaenh-serial-graph/onpolicy/scripts/train_smac_scripts/train_smac_${map}_SMD.sh"

    if [ ! -f "$script" ]; then
        echo "✗ $map: 脚本不存在"
        continue
    fi

    # 语法检查
    if ! bash -n "$script" 2>/dev/null; then
        echo "✗ $map: 脚本语法错误"
        continue
    fi

    # 干运行测试（不实际执行训练）
    echo "测试地图: $map"
    echo "  脚本路径: $script"

    # 使用 DRY_RUN 模式测试脚本
    if DRY_RUN=1 USE_WANDB=0 bash "$script" 2>&1 | head -5; then
        echo "  ✓ 脚本可正常运行"
    else
        echo "  ✗ 脚本执行失败"
    fi
    echo
done

echo "======================================================================"
echo "通用脚本测试"
echo "======================================================================"
echo

GENERIC_SCRIPT="on-policy-main-immaenh-serial-graph/onpolicy/scripts/train_smac_scripts/train_smac_SMD.sh"

if [ -f "$GENERIC_SCRIPT" ]; then
    echo "测试通用脚本: train_smac_SMD.sh"
    echo

    # 测试不同地图
    for map in "3m" "MMM" "corridor"; do
        echo "  使用地图: $map"
        if DRY_RUN=1 USE_WANDB=0 bash "$GENERIC_SCRIPT" "$map" 2>&1 | grep -q "map=${map}"; then
            echo "  ✓ 通用脚本支持 $map"
        else
            echo "  ✗ 通用脚本不支持 $map"
        fi
    done
    echo
else
    echo "✗ 通用脚本不存在"
fi

echo "======================================================================"
echo "测试完成"
echo "======================================================================"
echo
echo "要实际运行训练，请使用："
echo "  cd on-policy-main-immaenh-serial-graph/onpolicy/scripts/train_smac_scripts"
echo "  ./train_smac_3m_SMD.sh              # 训练 3m 地图"
echo "  ./train_smac_SMD.sh 8m              # 使用通用脚本训练 8m"
echo
echo "要自定义参数，请使用环境变量："
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  export NUM_ENV_STEPS=1000000        # 减少训练步数用于测试"
echo "  export ROLLOUT_THREADS=4"
echo "  ./train_smac_3m_SMD.sh"
echo
