"""Diagnose the D1 6.0 run: final values of each reward term + velocity-tracking
error, to see what's dominating (why it won't walk)."""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else max(
    glob.glob(os.path.expanduser("~/robot_lab6/logs/rsl_rl/navbot_d1_rough/*/")), key=os.path.getmtime
)
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])


def last(tag):
    s = ea.Scalars(tag)
    return s[-1].value if s else None


def show(prefix):
    rows = [(t, last(t)) for t in tags if t.startswith(prefix)]
    rows.sort(key=lambda r: (r[1] if r[1] is not None else 0))
    for t, v in rows:
        print(f"  {v:+9.4f}  {t}")


print(f"RUN: {os.path.basename(run.rstrip('/'))}")
print(f"final Train/mean_reward: {last('Train/mean_reward')}")
print(f"final Train/mean_episode_length: {last('Train/mean_episode_length')}")
print("\n--- Episode reward terms (most negative first) ---")
show("Episode_Reward/")
print("\n--- Metrics (velocity tracking error etc.) ---")
show("Metrics/")
print("\n--- Curriculum ---")
show("Curriculum/")
