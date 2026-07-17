#!/bin/bash
# Create desktop launcher buttons for Isaac Sim 6.0 (full GUI) and the D1 Isaac Lab viewer.
set -e
U="$HOME"

# ---- launcher scripts ----
cat > "$U/launch_isaacsim.sh" <<'EOF'
#!/bin/bash
# Full Isaac Sim 6.0 editor GUI
export DISPLAY=:0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate robotlab6
export OMNI_KIT_ACCEPT_EULA=yes
cd "$HOME"
exec isaacsim isaacsim.exp.full.kit
EOF
chmod +x "$U/launch_isaacsim.sh"

cat > "$U/launch_d1_viewer.sh" <<'EOF'
#!/bin/bash
# Trained D1 walking in the Isaac Lab viewport (16 robots)
export DISPLAY=:0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate robotlab6
export OMNI_KIT_ACCEPT_EULA=yes
cd "$HOME/robot_lab6"
exec python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 --num_envs 16 --load_run 2026-07-16_22-08-32
EOF
chmod +x "$U/launch_d1_viewer.sh"

# ---- desktop entries ----
mkdir -p "$U/Desktop"
cat > "$U/Desktop/IsaacSim6.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Isaac Sim 6.0
Comment=Full Isaac Sim editor GUI (load USDs, build scenes)
Exec=$U/launch_isaacsim.sh
Icon=applications-science
Terminal=true
Categories=Development;Science;
EOF

cat > "$U/Desktop/D1_IsaacLab_Walk.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=D1 - Isaac Lab Walk
Comment=Watch the trained D1 walking in the Isaac Lab viewport
Exec=$U/launch_d1_viewer.sh
Icon=applications-games
Terminal=true
Categories=Development;Science;
EOF

chmod +x "$U/Desktop/"*.desktop

# ---- mark trusted for the GNOME desktop ----
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"
for f in "$U/Desktop/IsaacSim6.desktop" "$U/Desktop/D1_IsaacLab_Walk.desktop"; do
  if gio set "$f" metadata::trusted true 2>/dev/null; then echo "trusted: $(basename "$f")"; else echo "trust-skip: $(basename "$f") (may need right-click > Allow Launching)"; fi
done
echo "=== DONE ==="
ls -la "$U/Desktop/"*.desktop
