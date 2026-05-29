"""
Schematic figure showing the routine signature workflow:
A. Input behavioral data (multi-channel time series)
B. GMM cluster assignment probabilities (heatmap)
C. Expected number of days per cluster (bar chart)
D. Routine signature (ranked proportions)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Reproducibility
rng = np.random.default_rng(42)

# ---------- Mock data generation ----------

# Panel A: 7 days of behavioral time series with sub-day resolution
n_days = 7
points_per_day = 50
t = np.linspace(1, n_days + 1, n_days * points_per_day)

def smooth_noise(scale, freq_mix, base=0.0, amp=1.0, seed=0):
    """Generate a smooth, somewhat periodic signal."""
    r = np.random.default_rng(seed)
    sig = base + amp * (
        0.5 * np.sin(2 * np.pi * freq_mix * t + r.uniform(0, 2 * np.pi))
        + 0.3 * np.sin(2 * np.pi * freq_mix * 2.3 * t + r.uniform(0, 2 * np.pi))
    )
    sig += r.normal(0, scale, size=t.shape)
    # Light smoothing
    kernel = np.ones(5) / 5
    sig = np.convolve(sig, kernel, mode='same')
    return sig

sleep_signal = smooth_noise(0.08, 0.5, base=0.55, amp=0.45, seed=1)
sleep_signal = np.clip(sleep_signal, 0, 1.2)

screen_signal = np.abs(smooth_noise(0.25, 1.8, base=0.05, amp=0.15, seed=2))
# Add sharp spikes
spike_idx = rng.choice(len(t), size=25, replace=False)
screen_signal[spike_idx] += rng.uniform(0.2, 0.4, size=25)
screen_signal = np.clip(screen_signal, 0, 0.55)

activity_signal = np.abs(smooth_noise(0.22, 1.5, base=0.05, amp=0.12, seed=3))
spike_idx = rng.choice(len(t), size=20, replace=False)
activity_signal[spike_idx] += rng.uniform(0.15, 0.35, size=20)
activity_signal = np.clip(activity_signal, 0, 0.55)

hr_signal = 0.4 + 0.15 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.05, size=t.shape)
kernel = np.ones(5) / 5
hr_signal = np.convolve(hr_signal, kernel, mode='same')
hr_signal = np.clip(hr_signal, 0, 1.0)

# Panel B: GMM cluster assignment probabilities
# 8 person-days x 8 clusters, mostly diagonal-ish with some spread
K = 8
n_rows = 8
prob_matrix = np.zeros((n_rows, K))
for i in range(n_rows):
    # Main mass on diagonal, smaller mass on neighbors
    probs = rng.dirichlet(np.ones(K) * 0.3)
    # Boost diagonal
    probs = probs * 0.3
    probs[i] += 0.7
    probs = probs / probs.sum()
    prob_matrix[i] = probs

# Panel C: expected number of days per cluster (decreasing)
expected_days = np.array([20, 15, 8.7, 5.4, 3.1, 1.7, 1.0, 0.9])

# Panel D: routine signature (proportions, decreasing)
signature = np.array([0.43, 0.285, 0.145, 0.075, 0.03, 0.015, 0.018, 0.015])

# ---------- Figure ----------

fig = plt.figure(figsize=(11, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
              left=0.08, right=0.95, top=0.95, bottom=0.07)

# ===== Panel A: Input Behavioral Data =====
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor('#ebebf0')

signals = [
    (sleep_signal, '#7FB3D5', 'Sleep Time', (0, 1.2)),
    (screen_signal, '#52BE80', 'Screen Use', (0, 0.55)),
    (activity_signal, '#9B59B6', 'Activity Levels', (0, 0.55)),
    (hr_signal, '#E74C3C', 'Heart Rate', (0, 1.0)),
]

# Stack the four time series as offsets within one axes
n_signals = len(signals)
panel_height = 1.0  # vertical space per signal in axis units
y_offsets = [3 * panel_height, 2 * panel_height, 1 * panel_height, 0]

handles = []
labels = []
for (sig, color, lbl, ylim), y_off in zip(signals, y_offsets):
    # Normalize each signal to fit in its band
    norm_sig = (sig - ylim[0]) / (ylim[1] - ylim[0]) * 0.85 + y_off
    h, = ax_a.plot(t, norm_sig, color=color, linewidth=1.2, label=lbl)
    handles.append(h)
    labels.append(lbl)

ax_a.set_xlim(1, n_days + 1)
ax_a.set_ylim(-0.1, 4.0)
ax_a.set_xticks(range(1, n_days + 1))
ax_a.set_xlabel('Days', fontsize=11)

# Custom y-axis: show 0.0 and 1.0/0.5 for each band
ytick_positions = []
ytick_labels = []
for y_off, (_, _, _, ylim) in zip(y_offsets, signals):
    ytick_positions.extend([y_off, y_off + 0.85])
    top = '1.0' if ylim[1] >= 1.0 else '0.5'
    ytick_labels.extend(['0.0', top])

ax_a.set_yticks(ytick_positions)
ax_a.set_yticklabels(ytick_labels, fontsize=8)
ax_a.set_ylabel('Normalized levels', fontsize=11)
ax_a.grid(True, color='white', linewidth=1.2, alpha=0.9)
ax_a.set_axisbelow(True)
for spine in ax_a.spines.values():
    spine.set_visible(False)
ax_a.tick_params(axis='both', length=0)

ax_a.legend(handles, labels, loc='upper right', fontsize=9,
            framealpha=0.95, edgecolor='gray')

ax_a.set_title('A.   Input Behavioral Data', loc='left',
               fontsize=14, fontweight='bold', pad=10)

# ===== Panel B: GMM Cluster Assignment Probabilities =====
ax_b = fig.add_subplot(gs[0, 1])
im = ax_b.imshow(prob_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax_b.set_xticks(range(K))
ax_b.set_xticklabels(range(1, K + 1), fontsize=10)
ax_b.set_yticks([])
ax_b.set_xlabel('K Clusters', fontsize=11)
ax_b.set_ylabel('Person-Days', fontsize=11)
ax_b.set_title('B.   GMM Cluster Assignment\n        Probabilities',
               loc='left', fontsize=14, fontweight='bold', pad=10)

# Colorbar
cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
cbar.set_label('Probability (0 to 1)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

for spine in ax_b.spines.values():
    spine.set_visible(False)

# ===== Panel C: Expected Number of Days per Cluster =====
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor('#ebebf0')

# Viridis-like gradient: purple at left -> green at right
# Use viridis with values constrained to 0.0-0.7 to avoid the yellow end
cmap = plt.get_cmap('viridis')
colors = [cmap(0.05 + 0.65 * (i / (K - 1))) for i in range(K)]

bars = ax_c.bar(range(1, K + 1), expected_days, color=colors,
                edgecolor='none', width=0.85)

ax_c.set_xticks(range(1, K + 1))
ax_c.set_xticklabels(range(1, K + 1), fontsize=10)
ax_c.set_xlabel('K Clusters', fontsize=11)
ax_c.set_ylabel('Expected Days', fontsize=11)
ax_c.set_ylim(0, 22)
ax_c.set_yticks([0, 5, 10, 15, 20])
ax_c.grid(True, axis='y', color='white', linewidth=1.2)
ax_c.set_axisbelow(True)
for spine in ax_c.spines.values():
    spine.set_visible(False)
ax_c.tick_params(axis='both', length=0)
ax_c.set_title('C.   Expected Number of Days per Cluster',
               loc='left', fontsize=14, fontweight='bold', pad=10)

# ===== Panel D: Routine Signature =====
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor('#ebebf0')

ax_d.scatter(range(1, K + 1), signature, s=60, color='#3B6FA8',
             edgecolor='none', zorder=3)

ax_d.set_xticks(range(1, K + 1))
ax_d.set_xticklabels(range(1, K + 1), fontsize=10)
ax_d.set_xlabel('cluster', fontsize=11)
ax_d.set_ylabel('proportion', fontsize=11)
ax_d.set_ylim(-0.02, 0.5)
ax_d.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax_d.grid(True, color='white', linewidth=1.2)
ax_d.set_axisbelow(True)
for spine in ax_d.spines.values():
    spine.set_visible(False)
ax_d.tick_params(axis='both', length=0)
ax_d.set_title('D.   Routine Signature',
               loc='left', fontsize=14, fontweight='bold', pad=10)

# ---------- Save ----------
out_path = '/mnt/user-data/outputs/routine_signature_schematic.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/routine_signature_schematic.pdf',
            bbox_inches='tight', facecolor='white')
print(f"Saved to {out_path}")
