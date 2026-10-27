# -*- coding: utf-8 -*-
"""生成 Dalamagkidis 致命概率随撞击动能变化的 S 形趋势图 (PNG, 300dpi)。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体：优先常见 Windows 字体，找不到就用默认 ----
for _name in ["Microsoft YaHei", "SimHei", "SimSun", "DengXian"]:
    if any(_name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- Dalamagkidis 公式参数 ----
alpha = 3.2e4     # 遮蔽概率 0.5 时达 50% 致死所需动能 (J)
beta  = 34.0      # 无遮蔽致命动能阈值 (J)
ps    = 0.25      # 遮蔽概率
K = np.sqrt(alpha / beta)

Ek = np.logspace(0, 6, 400)                        # 1 .. 1e6 J
P  = 1.0 / (1.0 + K * (beta / Ek) ** (1.0 / (4.0 * ps)))

fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(Ek, P, color="#2a78d6", lw=2.0, solid_capstyle="round", zorder=3)

ax.set_xscale("log")
ax.set_xlim(1, 1e6)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("撞击动能  $E_k$  (J)", fontsize=11, color="#52514e")
ax.set_ylabel("地面致命概率  $P_{fatality}$", fontsize=11, color="#52514e")
ax.set_yticks(np.arange(0, 1.01, 0.2))

# 学术风格描边
ax.grid(True, which="major", color="#e1e0d9", lw=0.8, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors="#898781", labelsize=9)

fig.tight_layout()
out = "fatality_vs_kinetic_energy.png"
fig.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved:", out)
