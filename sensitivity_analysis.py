#!/usr/bin/env python3
"""
Sensitivity analysis for tube-tilt eye-beam replication experiment.

Sweeps angle-acceptance cutoffs (low: >0…>12, high: <25…<50) and produces
heatmaps of D (angular deviation toward-minus-away) and P (one-sample t-test)
for each face individually, combined, and Weak vs Strong comparisons.

Trials are balanced: only matched pairs (same user, face, tube shape, face
side) with one towards and one away trial, both within the cutoff range.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import TwoSlopeNorm
import os

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'facetip_data_Nov2025.csv')
IMG_DIR = os.path.join(ROOT, 'face_images')
OUT_DIR = ROOT

FACES = ['ID015', 'ID017', 'ID030']
LO = np.arange(0, 13)
HI = np.arange(25, 51)
MIN_N = 10

FACE_LABELS = {
    'ID015': 'Non-threatening (glasses)',
    'ID017': 'Scary (tattooed)',
    'ID030': 'Scary (scarred)',
}

# ═══════════════════════════════════════════════════════════════════════
# Load face images
# ═══════════════════════════════════════════════════════════════════════
face_imgs = {}
face_imgs_right = {}
for fid in FACES:
    face_imgs[fid] = plt.imread(os.path.join(IMG_DIR, f'left_threatlevel_{fid}.png'))
    face_imgs_right[fid] = plt.imread(os.path.join(IMG_DIR, f'right_threatlevel_{fid}.png'))

# ═══════════════════════════════════════════════════════════════════════
# Load data & build balanced pairs
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA)
print(f"Loaded {len(df)} trials from {df['user_number'].nunique()} users")

recs = []
for (u, f, t, s), g in df.groupby(['user_number', 'face_id', 'tubeTypeIndex', 'faceSide']):
    tw = g[g['towards_away'] == 'towards']
    aw = g[g['towards_away'] == 'away']
    if len(tw) >= 1 and len(aw) >= 1:
        recs.append(dict(user=u, face=f, tube=t, side=s,
                         ang_t=tw['angle'].iloc[0], ang_a=aw['angle'].iloc[0]))

pairs = pd.DataFrame(recs)
print(f"Built {len(pairs)} balanced pairs")

# ═══════════════════════════════════════════════════════════════════════
# Sweep cutoffs
# ═══════════════════════════════════════════════════════════════════════
nL, nH = len(LO), len(HI)

face_groups = {
    'ID015': ['ID015'],
    'ID017': ['ID017'],
    'ID030': ['ID030'],
    'Combined': FACES,
}

comparisons = {
    'Scary1 vs Non-threat': ('ID017', 'ID015'),
    'Scary2 vs Non-threat': ('ID030', 'ID015'),
    'All Scary vs Non-threat': (['ID017', 'ID030'], 'ID015'),
}

mk = lambda keys: {k: np.full((nL, nH), np.nan) for k in keys}

gD, gP, gN, gNp = mk(face_groups), mk(face_groups), mk(face_groups), mk(face_groups)
cD, cP, cN = mk(comparisons), mk(comparisons), mk(comparisons)

# Total possible pairs per face (for percentage display)
total_pairs_per_face = pairs.groupby('face').size().to_dict()
total_pairs_all = len(pairs)

print("Sweeping cutoffs...", end="", flush=True)
for i, lo in enumerate(LO):
    for j, hi in enumerate(HI):
        valid = pairs[
            (pairs.ang_t > lo) & (pairs.ang_t < hi) &
            (pairs.ang_a > lo) & (pairs.ang_a < hi)
        ].copy()
        if valid.empty:
            continue
        valid['d'] = valid.ang_t - valid.ang_a

        uf = valid.groupby(['user', 'face'])['d'].mean()
        active_faces = uf.index.get_level_values('face').unique()

        # ── Individual faces + Combined ──
        for gname, flist in face_groups.items():
            if gname == 'Combined':
                face_valid = valid[valid.face.isin(flist)]
                uD = face_valid.groupby('user')['d'].mean()
                n_pairs = len(face_valid)
            else:
                fid = flist[0]
                if fid not in active_faces:
                    continue
                uD = uf.xs(fid, level='face')
                n_pairs = len(valid[valid.face == fid])
            if len(uD) > 1:
                t_s, p_v = stats.ttest_1samp(uD.values, 0)
                gD[gname][i, j] = uD.mean()
                gP[gname][i, j] = p_v
                gN[gname][i, j] = len(uD)
                gNp[gname][i, j] = n_pairs

        # ── Weak vs Strong comparisons ──
        for cn, (scary, weak) in comparisons.items():
            scary_l = [scary] if isinstance(scary, str) else scary
            if weak not in active_faces:
                continue
            Dw = uf.xs(weak, level='face')
            se = uf[uf.index.get_level_values('face').isin(scary_l)]
            if se.empty:
                continue
            Ds = se.groupby('user').mean()
            common = Dw.index.intersection(Ds.index)
            if len(common) > 1:
                diff = Ds.loc[common].values - Dw.loc[common].values
                t_s, p_v = stats.ttest_1samp(diff, 0)
                cD[cn][i, j] = diff.mean()
                cP[cn][i, j] = p_v
                cN[cn][i, j] = len(common)

    print(".", end="", flush=True)
print(" done.")

# ═══════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════
EXTENT = [HI[0] - 0.5, HI[-1] + 0.5, LO[0] - 0.5, LO[-1] + 0.5]


def plot_d_heatmap(ax, data, n_data, vmin, vmax, title=''):
    masked = np.where(n_data >= MIN_N, data, np.nan)
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    im = ax.imshow(masked, origin='lower', aspect='auto', extent=EXTENT,
                   cmap='RdBu_r', norm=norm, interpolation='none')
    ax.set_xlabel('Max angle cutoff (<)')
    ax.set_ylabel('Min angle cutoff (>)')
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks([25, 30, 35, 40, 45, 50])
    ax.set_yticks(range(0, 13, 2))
    return im


def plot_p_heatmap(ax, data, n_data, vmax_logp, title=''):
    masked_p = np.where(n_data >= MIN_N, data, np.nan)
    log_p = -np.log10(np.clip(masked_p, 1e-20, 1))
    im = ax.imshow(log_p, origin='lower', aspect='auto', extent=EXTENT,
                   cmap='inferno', vmin=0, vmax=vmax_logp,
                   interpolation='none')
    # p = 0.05 contour
    try:
        ax.contour(
            np.linspace(HI[0], HI[-1], nH),
            np.linspace(LO[0], LO[-1], nL),
            log_p, levels=[-np.log10(0.05)],
            colors='cyan', linewidths=1.5, linestyles='--')
    except ValueError:
        pass
    ax.set_xlabel('Max angle cutoff (<)')
    ax.set_ylabel('Min angle cutoff (>)')
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks([25, 30, 35, 40, 45, 50])
    ax.set_yticks(range(0, 13, 2))
    return im


def plot_pct_heatmap(ax, data, total, cmap, title=''):
    pct = data / total * 100 if total > 0 else data * 0
    im = ax.imshow(pct, origin='lower', aspect='auto', extent=EXTENT,
                   cmap=cmap, vmin=0, vmax=100, interpolation='none')
    for yi in range(0, nL, 3):
        for xi in range(0, nH, 5):
            val = pct[yi, xi]
            if not np.isnan(val):
                ax.text(HI[xi], LO[yi], f'{val:.0f}%',
                        ha='center', va='center', fontsize=6,
                        color='white' if val < 50 else 'black')
    ax.set_xlabel('Max angle cutoff (<)')
    ax.set_ylabel('Min angle cutoff (>)')
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks([25, 30, 35, 40, 45, 50])
    ax.set_yticks(range(0, 13, 2))
    return im


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Individual faces + Combined
#   Layout: 4 rows × 5 cols  (Face | D | P | N_pairs | N_subjects)
# ═══════════════════════════════════════════════════════════════════════
all_d_vals = np.concatenate([gD[g].ravel() for g in face_groups])
all_d_vals = all_d_vals[~np.isnan(all_d_vals)]
d_abs = max(abs(np.nanpercentile(all_d_vals, 2)),
            abs(np.nanpercentile(all_d_vals, 98)))
d_vmin, d_vmax = -d_abs, d_abs

all_p_vals = np.concatenate([gP[g].ravel() for g in face_groups])
all_p_vals = all_p_vals[~np.isnan(all_p_vals)]
all_logp = -np.log10(np.clip(all_p_vals[all_p_vals > 0], 1e-20, 1))
logp_vmax = min(np.nanpercentile(all_logp, 98), 6)

all_np_vals = np.concatenate([gNp[g].ravel() for g in face_groups])
all_np_vals = all_np_vals[~np.isnan(all_np_vals)]
np_vmax = np.nanmax(all_np_vals) if len(all_np_vals) > 0 else 100

all_ns_vals = np.concatenate([gN[g].ravel() for g in face_groups])
all_ns_vals = all_ns_vals[~np.isnan(all_ns_vals)]
ns_vmax = np.nanmax(all_ns_vals) if len(all_ns_vals) > 0 else 100

row_order = ['ID015', 'ID017', 'ID030', 'Combined']

fig1 = plt.figure(figsize=(30, 20))
gs1 = gridspec.GridSpec(4, 5, hspace=0.35, wspace=0.22,
                        left=0.03, right=0.96, top=0.93, bottom=0.04,
                        width_ratios=[0.55, 1, 1, 1, 1])

fig1.suptitle('Sensitivity Analysis: D, Significance, and Sample Size by Face',
              fontsize=16, fontweight='bold', y=0.97)

heatmap_axes_ns = []
heatmap_axes_np = []
heatmap_axes_d = []
heatmap_axes_p = []
d_im = p_im = ns_im = np_im = None

for row, gname in enumerate(row_order):
    label = FACE_LABELS.get(gname, 'All Faces Combined')
    total_for_face = (total_pairs_all if gname == 'Combined'
                      else total_pairs_per_face.get(gname, 0))
    total_subj = df[df['face_id'].isin(face_groups[gname])]['user_number'].nunique()

    # ── Col 0: Face photo ──
    ax_face = fig1.add_subplot(gs1[row, 0])
    ax_face.set_axis_off()
    if gname in face_imgs:
        ax_face.imshow(face_imgs[gname])
        ax_face.set_title(label, fontsize=10, fontweight='bold', pad=4)
    else:
        ax_face.text(0.5, 0.5, 'All Faces\nCombined',
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     transform=ax_face.transAxes)

    # ── Col 1: % Subjects ──
    ax_ns = fig1.add_subplot(gs1[row, 1])
    ns_im = plot_pct_heatmap(ax_ns, gN[gname], total_subj, 'YlOrRd',
                             title=f'% Subjects (N={total_subj})')
    heatmap_axes_ns.append(ax_ns)

    # ── Col 2: % Pairs ──
    ax_np = fig1.add_subplot(gs1[row, 2])
    np_im = plot_pct_heatmap(ax_np, gNp[gname], total_for_face, 'YlGnBu',
                             title=f'% Valid Pairs (N={total_for_face})')
    heatmap_axes_np.append(ax_np)

    # ── Col 3: D ──
    ax_d = fig1.add_subplot(gs1[row, 3])
    d_im = plot_d_heatmap(ax_d, gD[gname], gN[gname], d_vmin, d_vmax,
                          title='Mean D (°)')
    heatmap_axes_d.append(ax_d)

    # ── Col 4: P ──
    ax_p = fig1.add_subplot(gs1[row, 4])
    p_im = plot_p_heatmap(ax_p, gP[gname], gN[gname], logp_vmax,
                          title='Significance')
    heatmap_axes_p.append(ax_p)

fig1.colorbar(ns_im, ax=heatmap_axes_ns, shrink=0.7, pad=0.01,
              label='% of subjects')
fig1.colorbar(np_im, ax=heatmap_axes_np, shrink=0.7, pad=0.01,
              label='% of valid pairs')
fig1.colorbar(d_im, ax=heatmap_axes_d, shrink=0.7, pad=0.01,
              label='Mean D (degrees)')
cb_p = fig1.colorbar(p_im, ax=heatmap_axes_p, shrink=0.7, pad=0.01,
                     label='$-\\log_{10}(p)$')
cb_p.ax.axhline(-np.log10(0.05), color='cyan', linewidth=1.5, linestyle='--')
cb_p.ax.text(1.3, -np.log10(0.05), 'p=.05', va='center', fontsize=8,
             color='cyan', transform=cb_p.ax.get_yaxis_transform())

out1 = os.path.join(OUT_DIR, 'sensitivity_individual.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved {out1}")

# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Weak vs Strong comparisons
#   Layout: 3 rows × 3 cols  (Faces | D_diff | P)
# ═══════════════════════════════════════════════════════════════════════
all_cd = np.concatenate([cD[c].ravel() for c in comparisons])
all_cd = all_cd[~np.isnan(all_cd)]
if len(all_cd) > 0:
    cd_abs = max(abs(np.nanpercentile(all_cd, 2)),
                 abs(np.nanpercentile(all_cd, 98)))
else:
    cd_abs = 1.0
cd_vmin, cd_vmax = -cd_abs, cd_abs

all_cp = np.concatenate([cP[c].ravel() for c in comparisons])
all_cp = all_cp[~np.isnan(all_cp)]
if len(all_cp) > 0:
    all_clogp = -np.log10(np.clip(all_cp[all_cp > 0], 1e-20, 1))
    clogp_vmax = min(np.nanpercentile(all_clogp, 98), 6)
else:
    clogp_vmax = 3.0

comp_order = list(comparisons.keys())
comp_meta = {
    'Scary1 vs Non-threat': {'scary': 'ID017', 'weak': 'ID015'},
    'Scary2 vs Non-threat': {'scary': 'ID030', 'weak': 'ID015'},
    'All Scary vs Non-threat': {'scary': ['ID017', 'ID030'], 'weak': 'ID015'},
}

fig2 = plt.figure(figsize=(20, 16))
gs2 = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.25,
                        left=0.04, right=0.90, top=0.92, bottom=0.05,
                        width_ratios=[0.7, 1, 1])

fig2.suptitle('Sensitivity Analysis: Weak vs Strong (Paired Comparison)',
              fontsize=15, fontweight='bold', y=0.96)

comp_axes_d = []
comp_axes_p = []
cd_im = cp_im = None

for row, cname in enumerate(comp_order):
    scary, weak = comparisons[cname]
    meta = comp_meta[cname]
    n_range = cN[cname]
    n_min = int(np.nanmin(n_range)) if not np.all(np.isnan(n_range)) else 0
    n_max = int(np.nanmax(n_range)) if not np.all(np.isnan(n_range)) else 0

    # ── Face comparison panel ──
    ax_face = fig2.add_subplot(gs2[row, 0])
    ax_face.set_axis_off()
    scary_ids = meta['scary'] if isinstance(meta['scary'], list) else [meta['scary']]
    weak_id = meta['weak']

    n_faces = len(scary_ids) + 1
    face_size = 0.38 if n_faces <= 2 else 0.28

    if len(scary_ids) == 1:
        # Single scary vs weak: side by side with "vs" in the middle
        img_s = face_imgs[scary_ids[0]]
        img_w = face_imgs[weak_id]
        ax_s = ax_face.inset_axes([0.0, 0.15, face_size, 0.75])
        ax_s.imshow(img_s)
        ax_s.set_axis_off()
        ax_face.text(0.5, 0.5, 'vs', ha='center', va='center',
                     fontsize=14, fontstyle='italic', fontweight='bold',
                     transform=ax_face.transAxes)
        ax_w = ax_face.inset_axes([1.0 - face_size, 0.15, face_size, 0.75])
        ax_w.imshow(img_w)
        ax_w.set_axis_off()
    else:
        # Two scary faces vs weak
        y_top = 0.52
        y_bot = 0.0
        ax_s1 = ax_face.inset_axes([0.0, y_top, face_size, 0.48])
        ax_s1.imshow(face_imgs[scary_ids[0]])
        ax_s1.set_axis_off()
        ax_s2 = ax_face.inset_axes([0.0, y_bot, face_size, 0.48])
        ax_s2.imshow(face_imgs[scary_ids[1]])
        ax_s2.set_axis_off()
        ax_face.text(0.5, 0.5, 'vs', ha='center', va='center',
                     fontsize=14, fontstyle='italic', fontweight='bold',
                     transform=ax_face.transAxes)
        ax_w = ax_face.inset_axes([1.0 - face_size, 0.15, face_size, 0.7])
        ax_w.imshow(face_imgs[weak_id])
        ax_w.set_axis_off()

    # ── D_diff heatmap ──
    ax_d = fig2.add_subplot(gs2[row, 1])
    cd_im = plot_d_heatmap(ax_d, cD[cname], cN[cname], cd_vmin, cd_vmax,
                           title=f'Mean $D_{{diff}}$ (°)   [N: {n_min}–{n_max}]')
    comp_axes_d.append(ax_d)

    # ── P heatmap ──
    ax_p = fig2.add_subplot(gs2[row, 2])
    cp_im = plot_p_heatmap(ax_p, cP[cname], cN[cname], clogp_vmax,
                           title=f'Significance   [N: {n_min}–{n_max}]')
    comp_axes_p.append(ax_p)

if cd_im is not None:
    fig2.colorbar(cd_im, ax=comp_axes_d, shrink=0.7, pad=0.02,
                  label='Mean $D_{scary} - D_{weak}$ (degrees)')
if cp_im is not None:
    cb_cp = fig2.colorbar(cp_im, ax=comp_axes_p, shrink=0.7, pad=0.02,
                          label='$-\\log_{10}(p)$')
    cb_cp.ax.axhline(-np.log10(0.05), color='cyan', linewidth=1.5, linestyle='--')
    cb_cp.ax.text(1.3, -np.log10(0.05), 'p=.05', va='center', fontsize=8,
                  color='cyan', transform=cb_cp.ax.get_yaxis_transform())

out2 = os.path.join(OUT_DIR, 'sensitivity_comparisons.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved {out2}")

# ═══════════════════════════════════════════════════════════════════════
# Print summary at a representative cutoff
# ═══════════════════════════════════════════════════════════════════════
rep_lo, rep_hi = 3, 40
i_rep = np.searchsorted(LO, rep_lo)
j_rep = np.searchsorted(HI, rep_hi)

print(f"\n{'='*60}")
print(f"Reference cutoff: angle > {rep_lo} and angle < {rep_hi}")
print(f"{'='*60}")
for gname in row_order:
    d_val = gD[gname][i_rep, j_rep]
    p_val = gP[gname][i_rep, j_rep]
    n_val = gN[gname][i_rep, j_rep]
    sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
    print(f"  {gname:12s}  D={d_val:+.3f}°  p={p_val:.4f}  N={n_val:.0f}  {sig}")

print()
for cname in comp_order:
    d_val = cD[cname][i_rep, j_rep]
    p_val = cP[cname][i_rep, j_rep]
    n_val = cN[cname][i_rep, j_rep]
    sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
    print(f"  {cname:30s}  D_diff={d_val:+.3f}°  p={p_val:.4f}  N={n_val:.0f}  {sig}")

plt.close('all')
print("\nDone.")
