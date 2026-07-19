#!/bin/bash
# Isaac Sim 6.0.1 + Isaac Lab 3.0 + robot_lab on nuc1 (Blackwell RTX PRO 6000).
# Mirrors the p3tiny box: same recipe as migration_6.0/install_robotlab6.sh, but
# IsaacLab-3.0 and robot_lab6 are RSYNCED from the box (identical commits + the
# 9 platform fixes + staged navbot_d1) instead of cloned fresh.
# NOTE: driver 595.58.03+ is required for Isaac Sim 6 RTX rendering — nuc1 is on
# 580.x at install time; headless PhysX training may work, RTX needs the upgrade.
set -uo pipefail
LOG(){ echo "===== $(date +%H:%M:%S) $* ====="; }
BOX=p3tiny@192.168.1.197

LOG "[0/7] miniconda"
if [ ! -d ~/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh || { echo "MINICONDA_DL_FAIL"; exit 1; }
  bash /tmp/miniconda.sh -b -p ~/miniconda3 || { echo "MINICONDA_INSTALL_FAIL"; exit 1; }
  rm /tmp/miniconda.sh
fi
source ~/miniconda3/etc/profile.d/conda.sh
export OMNI_KIT_ACCEPT_EULA=yes

LOG "[1/7] create env robotlab6 (py3.12)"
if ! conda env list | grep -q '/robotlab6$'; then
  conda create -n robotlab6 python=3.12 -y -c conda-forge --override-channels || { echo "ENV_CREATE_FAIL"; exit 1; }
fi
conda activate robotlab6
python -V
PY="$CONDA_PREFIX/bin/python"

LOG "[2/7] install uv"
pip install -q uv || { echo "UV_FAIL"; exit 1; }

LOG "[3/7] install Isaac Sim 6.0.1.0 (big download)"
uv pip install --python "$PY" "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow || { echo "ISAACSIM_FAIL"; exit 2; }

LOG "[4/7] pin torch 2.11.0 cu128"
uv pip install --python "$PY" -U torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128 || { echo "TORCH_FAIL"; exit 3; }

LOG "[5/7] rsync IsaacLab-3.0 from the box (exact same commit)"
rsync -a --exclude 'logs/' --exclude 'outputs/' --exclude '_build/' \
  "$BOX:~/IsaacLab-3.0/" ~/IsaacLab-3.0/ || { echo "RSYNC_ISAACLAB_FAIL"; exit 4; }
cd ~/IsaacLab-3.0 && git log --oneline -1
./isaaclab.sh --install < /dev/null || { echo "ISAACLAB_INSTALL_FAIL"; exit 5; }

LOG "[6/7] rsync robot_lab6 from the box (9 fixes + staged navbot_d1)"
rsync -a --exclude 'logs/' "$BOX:~/robot_lab6/" ~/robot_lab6/ || { echo "RSYNC_ROBOTLAB_FAIL"; exit 6; }
uv pip install --python "$PY" -e ~/robot_lab6/source/robot_lab || { echo "ROBOTLAB_INSTALL_FAIL"; exit 7; }

LOG "[7/7] versions + imports"
$PY -c "import importlib.metadata as m; print('isaacsim', m.version('isaacsim'))" 2>&1 | tail -1
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1 | tail -1
$PY -c "import isaaclab; print('isaaclab OK')" 2>&1 | tail -1
$PY -c "import robot_lab; print('robot_lab OK')" 2>&1 | tail -1
echo "INSTALL_ALL_DONE"

# Post-install fixes discovered on first run (2026-07-19):
# 1. isaaclab.sh --install downgrades torch to 2.10 — re-pin AFTER it:
#    uv pip install --python "$PY" -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# 2. nuc1 has a user-site torch 2.10 in ~/.local that SHADOWS the env — block user-site:
#    conda env config vars set PYTHONNOUSERSITE=1 -n robotlab6
