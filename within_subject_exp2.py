#!/usr/bin/env python3
"""
Within-subject re-analysis of Exp 2 (threat faces).

Old approach:  Treat each face as a separate group → between-subject t-test of D vs 0.
New approach:  Each participant contributes D_per_face for EVERY face they saw.
               (1) One-sample t-test of each face's D vs 0 (same test, fuller data)
               (2) Paired t-test of D_high − D_low → direct threat modulation test,
                   eliminates between-subject noise.

Run:  python within_subject_exp2.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from schema_analysis.tube.load import load_from_json, mark_valid, balance_cascade
from schema_analysis.tube.treatments import resolve as resolve_treatment

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Load & clean ──────────────────────────────────────────────────────────────

ANGLE_LO, ANGLE_HI = 3, 40

df = load_from_json(os.path.join(ROOT, 'data', 'tube', 'raw', 'exp2'))
df = mark_valid(df, lo=ANGLE_LO, hi=ANGLE_HI)
df = balance_cascade(df, 'face_id')
valid = df[df['valid'] == True].copy()

faces = sorted(valid['face_id'].unique())  # ['ID015', 'ID030']
LOW_FACE = 'ID015'
HIGH_FACE = faces[-1]  # ID030 (or ID017 if run1 data present)

# ── Compute per-participant D for each face ──────────────────────────────────

rows = []
for uid, udata in valid.groupby('user_number'):
    row = {'user_number': uid}
    for fid in faces:
        sub = udata[udata['face_id'] == fid]
        tw = sub[sub['towards_away'] == 'towards']['angle']
        aw = sub[sub['towards_away'] == 'away']['angle']
        if len(tw) > 0 and len(aw) > 0:
            row[f'D_{fid}'] = tw.mean() - aw.mean()
        else:
            row[f'D_{fid}'] = np.nan
    rows.append(row)

ws = pd.DataFrame(rows)

# ── OLD (between-subject): includes anyone with at least one face ────────────

old_results = {}
for fid in faces:
    col = f'D_{fid}'
    d_vals = ws[col].dropna().values
    if len(d_vals) > 1:
        t, p = stats.ttest_1samp(d_vals, 0)
        old_results[fid] = dict(
            D=d_vals.mean(), SE=d_vals.std() / np.sqrt(len(d_vals)),
            t=t, p=p, N=len(d_vals),
        )

# ── NEW (within-subject): only participants with ALL faces ───────────────────

ws_complete = ws.dropna()
N_paired = len(ws_complete)

new_results = {}
for fid in faces:
    col = f'D_{fid}'
    d_vals = ws_complete[col].values
    t, p = stats.ttest_1samp(d_vals, 0)
    new_results[fid] = dict(
        D=d_vals.mean(), SE=d_vals.std() / np.sqrt(len(d_vals)),
        t=t, p=p, N=len(d_vals),
    )

# Paired difference: high − low
d_high = ws_complete[f'D_{HIGH_FACE}'].values
d_low = ws_complete[f'D_{LOW_FACE}'].values
d_diff = d_high - d_low
t_paired, p_paired = stats.ttest_rel(d_high, d_low)
paired = dict(
    D=d_diff.mean(), SE=d_diff.std() / np.sqrt(len(d_diff)),
    SD=d_diff.std(), t=t_paired, p=p_paired, N=N_paired,
)

# ── Console summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("WITHIN-SUBJECT RE-ANALYSIS — Exp 2 (Run 2: ID015 vs ID030)")
print("=" * 70)

print(f"\nAngle cutoffs: {ANGLE_LO}° < angle < {ANGLE_HI}°")
print(f"Participants with both faces: {N_paired}")

print("\n── One-sample t-tests (D vs 0) ──")
for fid in faces:
    r = new_results[fid]
    t_info = resolve_treatment(fid, False)
    sig = '*' if r['p'] < .05 else ''
    print(f"  {t_info['label']:25s}  N={r['N']:3d}  D={r['D']:+.4f}°  SE={r['SE']:.4f}  "
          f"t({r['N']-1})={r['t']:+.3f}  p={r['p']:.4f} {sig}")

print("\n── Paired t-test (threat modulation) ──")
print(f"  D_{HIGH_FACE} − D_{LOW_FACE} = {paired['D']:+.4f}°  SE={paired['SE']:.4f}")
print(f"  Within-subject SD = {paired['SD']:.4f}")
print(f"  t({paired['N']-1}) = {paired['t']:+.3f}  p = {paired['p']:.4f}")

print("\n── Comparison: old between-subject SEs ──")
for fid in faces:
    o = old_results.get(fid, {})
    n = new_results[fid]
    if o:
        reduction = (1 - n['SE'] / o['SE']) * 100
        print(f"  {fid}:  SE_old={o['SE']:.4f} (N={o['N']})  →  SE_new={n['SE']:.4f} (N={n['N']})  "
              f"({reduction:+.1f}%)")

# ── Plot ─────────────────────────────────────────────────────────────────────

IMG_DIR = os.path.join(ROOT, 'data', 'tube', 'images')

fig = plt.figure(figsize=(14, 7.5))
gs_top = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.0],
                            left=0.06, right=0.96, top=0.88, bottom=0.12,
                            wspace=0.35)

# ── Panel A: Between-subject (old) ──────────────────────────────────────────
ax_old = fig.add_subplot(gs_top[0])
for i, fid in enumerate(faces):
    r = old_results[fid]
    t_info = resolve_treatment(fid, False)
    ax_old.bar(i, r['D'], 0.55, color=t_info['color'], edgecolor='black', linewidth=0.8, alpha=0.5)
    ax_old.errorbar(i, r['D'], yerr=r['SE'], capsize=6, color='black', linewidth=1.5)
    if r['p'] < .05:
        ax_old.text(i, r['D'] + r['SE'] + 0.02, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    ax_old.text(i, -0.17, f"p={r['p']:.3f}\nN={r['N']}", ha='center', va='top', fontsize=8, color='#555',
                transform=ax_old.get_xaxis_transform(), clip_on=False)

ax_old.axhline(0, color='black', linewidth=0.8)
ax_old.set_xticks(range(len(faces)))
ax_old.set_xticklabels([resolve_treatment(f, False)['label'].replace('\n', ' ') for f in faces], fontsize=9)
ax_old.set_ylabel('D (degrees)')
ax_old.set_title('A. Between-subject\n(old analysis)', fontsize=11, fontweight='bold')
ax_old.spines['top'].set_visible(False)
ax_old.spines['right'].set_visible(False)

# ── Panel B: Within-subject (new) ───────────────────────────────────────────
ax_new = fig.add_subplot(gs_top[1])
for i, fid in enumerate(faces):
    r = new_results[fid]
    t_info = resolve_treatment(fid, False)
    ax_new.bar(i, r['D'], 0.55, color=t_info['color'], edgecolor='black', linewidth=0.8)
    ax_new.errorbar(i, r['D'], yerr=r['SE'], capsize=6, color='black', linewidth=1.5)
    if r['p'] < .05:
        ax_new.text(i, r['D'] + r['SE'] + 0.02, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    ax_new.text(i, -0.17, f"p={r['p']:.3f}\nN={r['N']}", ha='center', va='top', fontsize=8, color='#555',
                transform=ax_new.get_xaxis_transform(), clip_on=False)

ax_new.axhline(0, color='black', linewidth=0.8)
ax_new.set_xticks(range(len(faces)))
ax_new.set_xticklabels([resolve_treatment(f, False)['label'].replace('\n', ' ') for f in faces], fontsize=9)
ax_new.set_title('B. Within-subject\n(every participant → every face)', fontsize=11, fontweight='bold')
ax_new.spines['top'].set_visible(False)
ax_new.spines['right'].set_visible(False)

# Match y-axes
all_d = [old_results[f]['D'] for f in faces] + [new_results[f]['D'] for f in faces]
all_se = [old_results[f]['SE'] for f in faces] + [new_results[f]['SE'] for f in faces]
ylim = max(abs(d) + se for d, se in zip(all_d, all_se)) + 0.15
ylim = max(ylim, abs(paired['D']) + paired['SE'] + 0.15)
for ax in [ax_old, ax_new]:
    ax.set_ylim(-ylim, ylim)

# ── Panel C: Paired difference + distribution ────────────────────────────────
ax_paired = fig.add_subplot(gs_top[2])

# Violin / histogram of individual differences
ax_paired.hist(d_diff, bins=25, orientation='horizontal', color='#C06040', alpha=0.3,
               edgecolor='white', linewidth=0.5, density=True)
# Mean + SE bar
ax_paired.axhline(0, color='black', linewidth=0.8)
x_center = ax_paired.get_xlim()[1] * 0.5
ax_paired.plot(x_center, paired['D'], 'D', color='#C06040', markersize=14, zorder=5)
ax_paired.plot([x_center, x_center],
               [paired['D'] - paired['SE'], paired['D'] + paired['SE']],
               color='black', linewidth=2.5, zorder=4)
ax_paired.plot([x_center - 0.01, x_center + 0.01],
               [paired['D'] - paired['SE'], paired['D'] - paired['SE']],
               color='black', linewidth=2.5, zorder=4)
ax_paired.plot([x_center - 0.01, x_center + 0.01],
               [paired['D'] + paired['SE'], paired['D'] + paired['SE']],
               color='black', linewidth=2.5, zorder=4)

threat_label = resolve_treatment(HIGH_FACE, False)['label'].replace('\n', ' ')
low_label = resolve_treatment(LOW_FACE, False)['label'].replace('\n', ' ')
ax_paired.set_title(f'C. Paired difference\n(D_{{{HIGH_FACE}}} − D_{{{LOW_FACE}}})', fontsize=11, fontweight='bold')
ax_paired.set_ylabel(f'D difference (degrees)')
ax_paired.set_xlabel('Density')
ax_paired.set_ylim(-ylim * 2, ylim * 2)

# Annotate
txt = (f"Mean diff = {paired['D']:+.3f}°\n"
       f"SE = {paired['SE']:.3f}\n"
       f"t({paired['N']-1}) = {paired['t']:+.3f}\n"
       f"p = {paired['p']:.3f}\n"
       f"N = {paired['N']}")
ax_paired.text(0.95, 0.95, txt, ha='right', va='top', fontsize=9,
               transform=ax_paired.transAxes, family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8', edgecolor='#ccc'))

ax_paired.spines['top'].set_visible(False)
ax_paired.spines['right'].set_visible(False)

# ── Suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    f'Exp 2 Within-Subject Re-Analysis  |  Run 2: {LOW_FACE} (low) vs {HIGH_FACE} (high threat)\n'
    f'Angle cutoff {ANGLE_LO}°–{ANGLE_HI}° | N={N_paired} paired participants | '
    f'Error bars = SE | * p < .05',
    fontsize=10.5, y=0.97,
)

out = os.path.join(ROOT, 'within_subject_exp2.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nSaved {out}")
plt.close()
