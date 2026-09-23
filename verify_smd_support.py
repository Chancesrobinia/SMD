#!/usr/bin/env python3
"""验证22张标准SMAC地图是否都支持SMD方法训练"""

import os
import subprocess
from pathlib import Path

# 22张标准SMAC地图
STANDARD_MAPS = [
    "3m", "8m", "25m",
    "5m_vs_6m", "8m_vs_9m", "10m_vs_11m", "27m_vs_30m",
    "MMM", "MMM2",
    "2s3z", "3s5z", "3s5z_vs_3s6z",
    "3s_vs_3z", "3s_vs_4z", "3s_vs_5z",
    "1c3s5z",
    "2m_vs_1z",
    "corridor",
    "6h_vs_8z",
    "2s_vs_1sc",
    "so_many_baneling",
    "bane_vs_bane",
    "2c_vs_64zg",
]

SCRIPT_DIR = Path(__file__).parent / "on-policy-main-immaenh-serial-graph" / "onpolicy" / "scripts" / "train_smac_scripts"

def check_map_smd_support(map_name):
    """检查某个地图是否支持SMD训练"""
    script_path = SCRIPT_DIR / f"train_smac_{map_name}_SMD.sh"

    if not script_path.exists():
        return False, "脚本文件不存在"

    # 检查脚本语法
    try:
        subprocess.run(
            ["bash", "-n", str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return False, f"脚本语法错误: {e.stderr}"

    # 读取脚本内容检查关键参数
    content = script_path.read_text()

    # 如果脚本调用了通用的 train_smac_SMD.sh，则检查通用脚本
    if "train_smac_SMD.sh" in content and f'"{map_name}"' in content:
        generic_script = SCRIPT_DIR / "train_smac_SMD.sh"
        if not generic_script.exists():
            return False, "通用SMD脚本不存在"

        generic_content = generic_script.read_text()

        required_flags = [
            ("use_smd", "--use_smd"),
            ("use_hetero_graph", "--use_hetero_graph"),
            ("use_gated_fusion", "--use_gated_fusion"),
        ]

        missing = []
        for flag_name, flag_pattern in required_flags:
            if flag_pattern not in generic_content and f'"{flag_pattern}"' not in generic_content:
                missing.append(flag_name)

        if missing:
            return False, f"通用脚本缺少参数: {', '.join(missing)}"

        return True, "通过通用脚本支持"

    # 否则检查脚本本身
    required_flags = [
        ("use_smd", "--use_smd"),
        ("use_hetero_graph", "--use_hetero_graph"),
        ("use_gated_fusion", "--use_gated_fusion"),
    ]

    missing = []
    for flag_name, flag_pattern in required_flags:
        if flag_pattern not in content and f'"{flag_pattern}"' not in content:
            missing.append(flag_name)

    if missing:
        return False, f"缺少参数: {', '.join(missing)}"

    return True, "完全支持"

def main():
    print("=" * 70)
    print("验证22张标准SMAC地图的SMD方法支持情况")
    print("=" * 70)
    print()

    results = []
    supported_count = 0

    for map_name in STANDARD_MAPS:
        is_supported, message = check_map_smd_support(map_name)
        results.append((map_name, is_supported, message))
        if is_supported:
            supported_count += 1

    # 显示结果
    print(f"地图名称{' ' * 20}状态{' ' * 8}详情")
    print("-" * 70)

    for map_name, is_supported, message in results:
        status = "✓ 支持" if is_supported else "✗ 不支持"
        padding = " " * (28 - len(map_name))
        print(f"{map_name}{padding}{status:12s}{message}")

    print("-" * 70)
    print(f"\n总结: {supported_count}/{len(STANDARD_MAPS)} 张地图支持SMD方法训练")

    if supported_count == len(STANDARD_MAPS):
        print("\n🎉 恭喜！所有22张标准SMAC地图都已支持SMD方法训练！")
        return 0
    else:
        print(f"\n⚠️  还有 {len(STANDARD_MAPS) - supported_count} 张地图需要配置")
        return 1

if __name__ == "__main__":
    exit(main())
