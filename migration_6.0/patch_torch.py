"""Isaac Lab 3.0 port: robot_lab's custom mdp accesses physics-state buffers as if
they were torch tensors, but 3.0's Warp pipeline returns warp ProxyArrays whose
`.torch` view is required by the jit-scripted math functions. Add `.torch` to the
kinematic/joint buffers (matching stock isaaclab_tasks.core.velocity.mdp), leaving
config buffers (default_*) and sensor buffers alone (handled separately if needed)."""
import glob
import os
import re

MDP = os.path.expanduser(
    "~/robot_lab6/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp"
)
# physics-state warp buffers robot_lab reads (confirmed present in its mdp)
ATTRS = [
    "root_quat_w", "root_link_quat_w", "root_pos_w", "root_link_pos_w",
    "root_lin_vel_w", "root_lin_vel_b", "root_ang_vel_w", "root_ang_vel_b",
    "root_com_lin_vel_b", "body_pos_w", "body_link_pos_w", "body_lin_vel_w",
    "projected_gravity_b", "applied_torque", "joint_pos", "joint_vel",
]

total_files = 0
total_subs = 0
for f in sorted(glob.glob(os.path.join(MDP, "*.py"))):
    s = open(f).read()
    orig = s
    subs = 0
    for a in ATTRS:
        # match `.data.<attr>` NOT followed by an identifier char or a dot
        # (so `joint_pos` won't match `joint_pos_limits`, and won't double-add `.torch`)
        pat = r"\.data\." + a + r"(?![A-Za-z0-9_.])"
        s, n = re.subn(pat, ".data." + a + ".torch", s)
        subs += n
    if s != orig:
        open(f, "w").write(s)
        print(f"patched {os.path.basename(f)}: {subs} sites")
        total_files += 1
        total_subs += subs
print(f"TOTAL: {total_subs} sites in {total_files} files")
