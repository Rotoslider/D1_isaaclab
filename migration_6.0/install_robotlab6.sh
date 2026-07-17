#!/bin/bash
# Isaac Sim 6.0.1 + Isaac Lab 3.0 in an ISOLATED env (robotlab6, py3.12).
# Does NOT touch the working robotlab (5.1) env — that stays as the fallback.
set -uo pipefail
LOG(){ echo "===== $(date +%H:%M:%S) $* ====="; }
source ~/miniconda3/etc/profile.d/conda.sh
export OMNI_KIT_ACCEPT_EULA=yes

LOG "[1/6] create env robotlab6 (py3.12)"
if ! conda env list | grep -q '/robotlab6$'; then
  conda create -n robotlab6 python=3.12 -y -c conda-forge --override-channels || { echo "ENV_CREATE_FAIL"; exit 1; }
fi
conda activate robotlab6
python -V
PY="$CONDA_PREFIX/bin/python"

LOG "[2/6] install uv"
pip install -q uv || { echo "UV_FAIL"; exit 1; }

LOG "[3/6] install Isaac Sim 6.0.1.0 (big download, several GB)"
uv pip install --python "$PY" "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow || { echo "ISAACSIM_FAIL"; exit 2; }

LOG "[4/6] pin torch 2.11.0 cu128"
uv pip install --python "$PY" -U torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128 || { echo "TORCH_FAIL"; exit 3; }

LOG "[5/6] clone Isaac Lab (develop = 3.0)"
if [ ! -d ~/IsaacLab-3.0 ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git --branch develop ~/IsaacLab-3.0 || { echo "CLONE_FAIL"; exit 4; }
fi
cd ~/IsaacLab-3.0
git log --oneline -1

LOG "[6/6] isaaclab.sh --install"
./isaaclab.sh --install < /dev/null || { echo "ISAACLAB_INSTALL_FAIL"; exit 5; }

LOG "versions"
$PY -c "import isaacsim,importlib.metadata as m; print('isaacsim', m.version('isaacsim'))" 2>&1 | tail -1
$PY -c "import isaaclab; print('isaaclab OK')" 2>&1 | tail -1
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" 2>&1 | tail -1
echo "INSTALL_ALL_DONE"
