import os
import subprocess
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "onpolicy" / "scripts" / "train_smac_scripts"

STANDARD_MAP_WRAPPERS = {
    "2s3z": "train_smac_2s3z_SMD.sh",
    "8m": "train_smac_8m_SMD.sh",
    "25m": "train_smac_25m_SMD.sh",
    "5m_vs_6m": "train_smac_5m_vs_6m_SMD.sh",
    "8m_vs_9m": "train_smac_8m_vs_9m_SMD.sh",
    "10m_vs_11m": "train_smac_10m_vs_11m_SMD.sh",
    "27m_vs_30m": "train_smac_27m_vs_30m_SMD.sh",
    "MMM": "train_smac_MMM_SMD.sh",
    "MMM2": "train_smac_MMM2_SMD.sh",
    "3s5z": "train_smac_3s5z_SMD.sh",
    "3s5z_vs_3s6z": "train_smac_3s5z_vs_3s6z_SMD.sh",
    "3s_vs_3z": "train_smac_3s_vs_3z_SMD.sh",
    "3s_vs_4z": "train_smac_3s_vs_4z_SMD.sh",
    "3s_vs_5z": "train_smac_3s_vs_5z_SMD.sh",
    "1c3s5z": "train_smac_1c3s5z_SMD.sh",
    "2m_vs_1z": "train_smac_2m_vs_1z_SMD.sh",
    "corridor": "train_smac_corridor_SMD.sh",
    "6h_vs_8z": "train_smac_6h_vs_8z_SMD.sh",
    "2s_vs_1sc": "train_smac_2s_vs_1sc_SMD.sh",
    "so_many_baneling": "train_smac_so_many_baneling_SMD.sh",
    "bane_vs_bane": "train_smac_bane_vs_bane_SMD.sh",
    "2c_vs_64zg": "train_smac_2c_vs_64zg_SMD.sh",
}


def run_dry(script_name, *extra_args, **env_overrides):
    env = os.environ.copy()
    env.update({
        "DRY_RUN": "1",
        "USE_WANDB": "0",
        **env_overrides,
    })
    return subprocess.run(
        [str(SCRIPT_DIR / script_name), *extra_args],
        cwd=PROJECT_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.mark.parametrize("map_name,script_name", STANDARD_MAP_WRAPPERS.items())
def test_standard_map_wrapper_builds_smd_training_command(map_name, script_name):
    script = SCRIPT_DIR / script_name

    assert script.is_file()
    assert os.access(script, os.X_OK)

    output = run_dry(script_name)
    command = output.split("Command:", 1)[1]
    assert "--map_name {}".format(map_name) in command
    assert "--use_hetero_graph" in command
    assert "--use_gated_fusion" in command
    assert "--use_smd" in command
    assert "--use_stacked_frames" not in command


@pytest.mark.parametrize(
    "script_name,expected_args",
    [
        ("train_smac_MMM2_SMD.sh", ["--num_mini_batch 2", "--ppo_epoch 5", "--gain 1"]),
        ("train_smac_5m_vs_6m_SMD.sh", ["--ppo_epoch 10", "--clip_param 0.05"]),
        ("train_smac_3s_vs_5z_SMD.sh", ["--algorithm_name mappo", "--clip_param 0.05"]),
        ("train_smac_2m_vs_1z_SMD.sh", ["--smd_candidate_top_k 1"]),
    ],
)
def test_map_specific_training_settings_are_preserved(script_name, expected_args):
    command = run_dry(script_name).split("Command:", 1)[1]

    for expected in expected_args:
        assert expected in command


def test_generic_launcher_accepts_environment_and_cli_overrides():
    output = run_dry(
        "train_smac_SMD.sh",
        "8m",
        "--num_env_steps",
        "123",
        ROLLOUT_THREADS="2",
        SMD_EDGE_TOP_M="2",
        USE_EVAL="0",
    )
    command = output.split("Command:", 1)[1]

    assert "--n_rollout_threads 2" in command
    assert "--smd_edge_top_m 2" in command
    assert "--num_env_steps 123" in command
    assert "--use_eval" not in command
    assert "--use_wandb" in command

    wandb_command = run_dry(
        "train_smac_SMD.sh",
        "8m",
        USE_WANDB="1",
    ).split("Command:", 1)[1]
    assert "--use_wandb" not in wandb_command


def test_all_smd_scripts_have_valid_shell_syntax():
    scripts = sorted(SCRIPT_DIR.glob("*SMD*.sh"))

    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
