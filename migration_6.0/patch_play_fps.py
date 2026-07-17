"""Isaac Lab 3.0 port fix: our env doesn't advertise render_fps, so gymnasium's
RecordVideo passes fps=None to moviepy. Set it from the env's step_dt right
before the RecordVideo wrap in play.py."""
import os

p = os.path.expanduser("~/robot_lab6/scripts/reinforcement_learning/rsl_rl/play.py")
lines = open(p).readlines()
if any("render_fps" in l for l in lines):
    print("already patched")
else:
    for i, l in enumerate(lines):
        if "gym.wrappers.RecordVideo(env" in l:
            indent = l[: len(l) - len(l.lstrip())]
            fix = [
                indent + 'if env.metadata.get("render_fps") is None:\n',
                indent + '    env.metadata["render_fps"] = int(round(1.0 / env.unwrapped.step_dt))\n',
            ]
            lines[i:i] = fix
            open(p, "w").writelines(lines)
            print(f"patched play.py at line {i+1}")
            break
    else:
        print("RecordVideo line not found")
