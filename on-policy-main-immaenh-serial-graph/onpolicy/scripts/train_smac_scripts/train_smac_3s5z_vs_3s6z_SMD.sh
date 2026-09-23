#!/bin/sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${SCRIPT_DIR}/train_smac_SMD.sh" "3s5z_vs_3s6z" "$@"
