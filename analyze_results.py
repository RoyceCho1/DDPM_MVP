import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH = Path("/home/work/DDPM/DDPM_MVP/results/ddpm_cifar10_v1/train.log")
OUT_PATH = LOG_PATH.parent / "log_curve.png"

pattern = re.compile(r"Step\s+(\d+)\s+Loss:\s+([0-9.eE+-]+)\s+LR:\s+([0-9.eE+-]+)")

steps, losses, lrs = [], [], []

with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            lrs.append(float(m.group(3)))

if len(steps) == 0:
    raise RuntimeError("No matching log lines found. Check log format / regex.")

steps = np.array(steps)
losses = np.array(losses)
lrs = np.array(lrs)

def moving_average(x, w=200):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

# ----- Plot Loss -----
plt.figure()
plt.plot(steps, losses, linewidth=1)
ma_w = 200
if len(losses) > ma_w:
    plt.plot(steps[ma_w-1:], moving_average(losses, ma_w), linewidth=2)
plt.xlabel("Step")
plt.ylabel("Loss (MSE on eps)")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig(OUT_PATH.with_name("loss_curve.png"), dpi=200)
plt.close()

# ----- Plot LR -----
plt.figure()
plt.plot(steps, lrs, linewidth=1)
plt.xlabel("Step")
plt.ylabel("LR")
plt.title("Learning Rate")
plt.tight_layout()
plt.savefig(OUT_PATH.with_name("lr_curve.png"), dpi=200)
plt.close()

print("Parsed points:", len(steps))
print("Loss: min", losses.min(), "max", losses.max(), "last", losses[-1])
print("LR  : min", lrs.min(), "max", lrs.max(), "last", lrs[-1])
print("Saved:", OUT_PATH.with_name("loss_curve.png"))
print("Saved:", OUT_PATH.with_name("lr_curve.png"))
