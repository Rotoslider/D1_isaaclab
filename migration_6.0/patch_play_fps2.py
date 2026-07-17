"""Pass fps explicitly to RecordVideo (gymnasium 1.3.0 accepts it) — the metadata
route wasn't sticking, so set it in video_kwargs right before the wrapper."""
import os

p = os.path.expanduser("~/robot_lab6/scripts/reinforcement_learning/rsl_rl/play.py")
lines = open(p).readlines()
if any('video_kwargs["fps"]' in l for l in lines):
    print("already patched")
else:
    for i, l in enumerate(lines):
        if "gym.wrappers.RecordVideo(env" in l:
            indent = l[: len(l) - len(l.lstrip())]
            lines.insert(i, indent + 'video_kwargs["fps"] = int(round(1.0 / env.unwrapped.step_dt))\n')
            open(p, "w").writelines(lines)
            print(f"patched: video_kwargs['fps'] inserted at line {i+1}")
            break
    else:
        print("RecordVideo line not found")
