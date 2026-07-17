import os

p = os.path.expanduser("~/robot_lab6/source/robot_lab/robot_lab/assets/navbot_d1.py")
s = open(p).read()
start = s.index("joint_pos={")
end = s.index("}", start) + 1
seg = s[start:end]
new = (
    'joint_pos={\n'
    '            ".*L_hip_joint": -0.05,\n'
    '            ".*R_hip_joint": 0.05,\n'
    '            ".*_thigh_joint": -0.75,\n'
    '            ".*_calf_joint": -0.75,\n'
    '        }'
)
s = s.replace(seg, new)
for old_pos in ("pos=(0.0, 0.0, 0.40)", "pos=(0.0, 0.0, 0.55)"):
    s = s.replace(old_pos, "pos=(0.0, 0.0, 0.53)")
open(p, "w").write(s)
print("set Frank exact pose (-0.05/0.05, -0.75, -0.75) + spawn 0.53")
