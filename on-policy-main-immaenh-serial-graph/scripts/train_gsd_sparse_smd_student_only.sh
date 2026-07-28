#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_gsd_sparse_smd.sh" --disable_smd_diffusion_teacher --lambda_smd_diff 0.0
