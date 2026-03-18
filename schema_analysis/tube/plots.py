"""
Plots for tube-tilt experiments.

    DFigure(sessions).plot_d(save='figure_d_bars.png')  — D bar plot
    sensitivity_heatmap(sessions, save='sensitivity_exp1.png') — cutoff sweep heatmap
    angle_correlation(sessions, save='correlation_angles.png') — user vs critical angle
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .treatments import resolve as resolve_treatment
from .compute import compute_d, sweep_sensitivity

if TYPE_CHECKING:
    from .sessions import TubeSessions

# Canonical group display order
_EXP1_ORDER = [('ID008', False), ('ID008', True)]
_EXP2_ORDER = [('ID015', False), ('ID017', False)]


# ── D bar figure ───────────────────────────────────────────────────────────────

class DFigure:
    """
    D bar figure. Auto-discovers condition groups from sessions present.

    If only Exp1 sessions: 2 bars (eyes open / covered).
    If only Exp2 sessions: 2 bars (ID015 / ID017).
    If both: combined figure with spacer.
    """

    def __init__(self, sessions: "TubeSessions"):
        self._sessions = sessions

    def _discover_groups(self) -> Tuple[List[dict], bool, bool]:
        """
        Discover condition groups present in the sessions.

        Returns (groups, has_exp1, has_exp2).
        Each group: {face_id, eyes_covered, exp_num, run, label, color, image_path}
        """
        e1_present = set()
        e2_present = set()  # face_ids seen in Exp2

        for sess in self._sessions:
            if sess.exp_num == 1 and sess.face_id is not None:
                ec = bool(sess.eyes_covered) if sess.eyes_covered is not None else False
                e1_present.add((sess.face_id, ec))
            elif sess.exp_num == 2 and 'face_id' in sess.trials.columns:
                for fid in sess.trials['face_id'].unique():
                    e2_present.add(fid)

        groups = []

        # Exp1 in canonical order
        for key in _EXP1_ORDER:
            if key in e1_present:
                t = resolve_treatment(key[0], key[1])
                groups.append(dict(
                    face_id=key[0], eyes_covered=key[1], exp_num=1, run=None,
                    split_by='eyes_covered', value=key[1],
                    label=t['label'], color=t['color'], image_path=t['image_path'],
                ))

        # Exp2 in canonical order
        for fid, _ in _EXP2_ORDER:
            if fid in e2_present:
                t = resolve_treatment(fid, False)
                groups.append(dict(
                    face_id=fid, eyes_covered=False, exp_num=2,
                    split_by='face_id', value=fid,
                    label=t['label'], color=t['color'], image_path=t['image_path'],
                ))

        has_exp1 = any(g['exp_num'] == 1 for g in groups)
        has_exp2 = any(g['exp_num'] == 2 for g in groups)
        return groups, has_exp1, has_exp2

    def _build_bars(self, groups: List[dict]) -> List[Optional[dict]]:
        """Compute D for each group. Returns list (no spacer — divider is drawn separately)."""
        bars: List[Optional[dict]] = []
        for g in groups:
            # Filter to the experiment's sessions only so eyes_covered=False
            # in Exp2 doesn't bleed into the Exp1 "Eyes Open" group.
            sess = self._sessions.select(exp_num=g['exp_num'])
            result = compute_d(sess, g['split_by'], g['value'])
            bars.append({**g, 'result': result})
        return bars

    def plot_d(self, *, save: Optional[str] = None, show: bool = False) -> plt.Figure:
        """
        Render D bar chart. Discovers groups from sessions present.

        save='path.png' — save to file.
        show=True       — display interactively.
        """
        groups, has_exp1, has_exp2 = self._discover_groups()
        if not groups:
            print("  No groups found in sessions.")
            return None

        bars = self._build_bars(groups)
        fig = self._render(bars, has_exp1=has_exp1, has_exp2=has_exp2)

        if save:
            fig.savefig(save, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"Saved {save}")
        if show:
            plt.show()
        else:
            plt.close()
        return fig

    def _render(self, bars: List[Optional[dict]], *, has_exp1: bool, has_exp2: bool) -> plt.Figure:
        n_bars = len(bars)
        non_spacer = [(i, b) for i, b in enumerate(bars) if b is not None]

        fig = plt.figure(figsize=(max(10, n_bars * 2 + 2), 6.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[0.75, 1], hspace=0.01,
                               left=0.08, right=0.97, top=0.87, bottom=0.18)
        ax_img = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
        ax_img.set_axis_off()
        ax_img.set_xlim(-0.5, n_bars - 0.5)

        # Face images from treatments + N label below each face
        for i, b in non_spacer:
            img_path = b.get('image_path')
            r = b['result']
            if img_path and os.path.exists(img_path):
                img = plt.imread(img_path)
                # Normalise to same display height so all faces are consistent
                h, w = img.shape[:2]
                target_zoom = 0.18 * (500 / h)   # anchor to a 500-px-tall reference
                oi = OffsetImage(img, zoom=target_zoom)
                ab = AnnotationBbox(oi, (i, 0.55), frameon=False,
                                    xycoords=('data', 'axes fraction'),
                                    box_alignment=(0.5, 0.5))
                ax_img.add_artist(ab)
            # N label just below the face panel
            ax_img.text(i, 0.04, f'N={r["N"]}', ha='center', va='bottom',
                        fontsize=8, color='#444',
                        transform=ax_img.get_xaxis_transform())

        # Exp group labels and divider
        xlim_lo, xlim_hi = -0.6, n_bars - 0.4
        x_span = xlim_hi - xlim_lo

        def _bar_cx(exp_num):
            """Axes-fraction x of the centre of all bars for a given exp."""
            positions = [i for i, b in enumerate(bars)
                         if b is not None and b.get('exp_num') == exp_num]
            return (np.mean(positions) - xlim_lo) / x_span

        if has_exp1 and has_exp2:
            last_e1 = max(i for i, b in enumerate(bars) if b is not None and b['exp_num'] == 1)
            first_e2 = min(i for i, b in enumerate(bars) if b is not None and b['exp_num'] == 2)
            divider_x = (last_e1 + first_e2) / 2
            ln = ax_bar.axvline(x=divider_x, color='#bbbbbb',
                                linewidth=1.5, linestyle='--', zorder=0)
            ln.set_clip_on(False)
            ax_img.text(_bar_cx(1), 0.97, 'Exp 1: Eyes Open vs. Closed',
                        ha='center', va='top', fontsize=10, fontweight='bold',
                        transform=ax_img.transAxes, color='#333')
            ax_img.text(_bar_cx(2), 0.97, 'Exp 2: Threat Level',
                        ha='center', va='top', fontsize=10, fontweight='bold',
                        transform=ax_img.transAxes, color='#333')
        elif has_exp1:
            ax_img.text(0.5, 0.97, 'Exp 1: Eyes Open vs. Closed', ha='center', va='top',
                        fontsize=10, fontweight='bold', transform=ax_img.transAxes, color='#333')
        elif has_exp2:
            ax_img.text(0.5, 0.97, 'Exp 2: Threat Level', ha='center', va='top',
                        fontsize=10, fontweight='bold', transform=ax_img.transAxes, color='#333')

        # Bars
        for i, b in non_spacer:
            r = b['result']
            color = b.get('color', '#888888')
            ax_bar.bar(i, r['D'], 0.55, color=color, edgecolor='black', linewidth=0.8)
            ax_bar.errorbar(i, r['D'], yerr=r['SE'], capsize=5, color='black', linewidth=1.5)
            if r['p'] < 0.05:
                ax_bar.text(i, r['D'] + r['SE'] + 0.015, '*',
                            ha='center', va='bottom', fontsize=22, fontweight='bold')
            ax_bar.text(i, -0.18, f"p={r['p']:.3f}",
                        ha='center', va='top', fontsize=8, color='#555',
                        transform=ax_bar.get_xaxis_transform(),
                        clip_on=False)

        ax_bar.axhline(y=0, color='black', linewidth=0.8)
        ax_bar.set_xticks([i for i, _ in non_spacer])
        # Flatten multi-line labels to single line for cleaner tick display
        ax_bar.set_xticklabels(
            [b['label'].replace('\n', ' ') for _, b in non_spacer],
            fontsize=9,
        )
        ax_bar.set_xlim(-0.6, n_bars - 0.4)
        ax_bar.set_ylabel('D (degrees)')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        yvals = ([b['result']['D'] + b['result']['SE'] for _, b in non_spacer] +
                 [b['result']['D'] - b['result']['SE'] for _, b in non_spacer])
        ylim = max(abs(min(yvals)), abs(max(yvals))) + 0.10
        ax_bar.set_ylim(-ylim, ylim)

        from .load import ANGLE_LO, ANGLE_HI
        fig.suptitle(
            f'D = mean(towards angle) − mean(away angle)\n'
            f'Angle cutoff {ANGLE_LO}°–{ANGLE_HI}° | t-test vs 0 | * p < .05 | Error bars = SE',
            fontsize=11, y=0.97,
        )
        return fig


# ── Sensitivity heatmap ────────────────────────────────────────────────────────

def sensitivity_heatmap(
    sessions: "TubeSessions",
    *,
    save: Optional[str] = None,
    show: bool = False,
    title: str = '',
) -> plt.Figure:
    """
    Sweep angle cutoffs and plot D and significance heatmaps per condition.

    Each row = one condition (face_id + eyes_covered).
    Columns: face image, Mean D (heatmap), Significance (-log10 p).
    """
    sweep = sweep_sensitivity(sessions)
    if not sweep['groups']:
        print("  No sensitivity data.")
        return None

    groups = sweep['groups']
    group_labels = sweep['group_labels']
    lo = sweep['lo_range']
    hi = sweep['hi_range']
    min_n = sweep['min_n']

    # Order: Exp1 first, then Exp2
    ordered_keys = []
    for k in _EXP1_ORDER:
        if k in groups:
            ordered_keys.append(k)
    for fid, _ in _EXP2_ORDER:
        k = (fid, False)
        if k in groups:
            ordered_keys.append(k)
    # Fallback: any remaining
    for k in groups:
        if k not in ordered_keys:
            ordered_keys.append(k)

    if not ordered_keys:
        return None

    n_rows = len(ordered_keys)

    # Global D scale
    all_d = np.concatenate([groups[k]['D'].ravel() for k in ordered_keys])
    all_d = all_d[~np.isnan(all_d)]
    d_abs = max(abs(np.nanpercentile(all_d, 2)), abs(np.nanpercentile(all_d, 98))) if len(all_d) else 1.0

    # Global p scale
    all_p = np.concatenate([groups[k]['P'].ravel() for k in ordered_keys])
    all_p = all_p[~np.isnan(all_p)]
    logp_vmax = 3.0
    if len(all_p) > 0:
        valid_p = all_p[(all_p > 0) & ~np.isnan(all_p)]
        if len(valid_p) > 0:
            logp_vmax = min(np.nanpercentile(-np.log10(np.clip(valid_p, 1e-20, 1)), 98), 6)

    ext = [hi[0] - 0.5, hi[-1] + 0.5, lo[0] - 0.5, lo[-1] + 0.5]

    from .load import ANGLE_LO, ANGLE_HI  # noqa: F401 (kept for potential future use)

    # Dynamic ticks from actual sweep ranges (every 5 for hi, every 2 for lo)
    hi_step = max(1, (hi[-1] - hi[0]) // 6)
    lo_step = max(1, (lo[-1] - lo[0]) // 5)
    hi_ticks = hi[::hi_step]
    lo_ticks = lo[::lo_step]

    fig = plt.figure(figsize=(16, 3.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, hspace=0.55, wspace=0.18,
                           left=0.07, right=0.93, top=0.94, bottom=0.08,
                           width_ratios=[0.35, 1, 1])

    fig_title = title or 'Sensitivity Analysis — Angle Cutoff Sweep'
    fig.suptitle(fig_title, fontsize=13, fontweight='bold', y=0.98)

    d_axes, p_axes = [], []
    d_im = p_im = None

    for row, key in enumerate(ordered_keys):
        fid, ec = key
        t = resolve_treatment(fid, ec)
        g = groups[key]
        n_data = g['N']
        d_data = np.where(n_data >= min_n, g['D'], np.nan)
        p_data = np.where(n_data >= min_n, g['P'], np.nan)
        log_p = -np.log10(np.clip(p_data, 1e-20, 1))

        # Col 0: face image or label + N at standard cutoff
        ax_face = fig.add_subplot(gs[row, 0])
        ax_face.set_axis_off()
        img_path = t.get('image_path')
        n_std = int(np.nanmax(g['N'])) if not np.all(np.isnan(g['N'])) else 0
        if img_path and os.path.exists(img_path):
            ax_face.imshow(plt.imread(img_path))
            ax_face.set_title(t['label'], fontsize=10, fontweight='bold', pad=4)
        else:
            ax_face.text(0.5, 0.5, t['label'], ha='center', va='center',
                         fontsize=12, fontweight='bold', transform=ax_face.transAxes)
        ax_face.text(0.5, -0.08, f'N={n_std}', ha='center', va='top',
                     fontsize=9, color='#333', transform=ax_face.transAxes,
                     clip_on=False)

        # Col 1: Mean D heatmap
        ax_d = fig.add_subplot(gs[row, 1])
        norm = TwoSlopeNorm(vcenter=0, vmin=-d_abs, vmax=d_abs)
        d_im = ax_d.imshow(d_data, origin='lower', aspect='auto', extent=ext,
                           cmap='RdBu_r', norm=norm, interpolation='none')
        ax_d.set_xlabel('Max angle cutoff (<)', fontsize=9)
        ax_d.set_ylabel('Min angle cutoff (>)', fontsize=9)
        ax_d.set_title('Mean D (°)', fontsize=10, pad=4)
        ax_d.set_xticks(hi_ticks)
        ax_d.set_yticks(lo_ticks)
        ax_d.tick_params(labelsize=8)
        d_axes.append(ax_d)

        # Col 2: Significance heatmap
        ax_p = fig.add_subplot(gs[row, 2])
        p_im = ax_p.imshow(log_p, origin='lower', aspect='auto', extent=ext,
                           cmap='inferno', vmin=0, vmax=logp_vmax, interpolation='none')
        nL, nH = len(lo), len(hi)
        try:
            ax_p.contour(
                np.linspace(hi[0], hi[-1], nH),
                np.linspace(lo[0], lo[-1], nL),
                log_p, levels=[-np.log10(0.05)],
                colors='cyan', linewidths=1.5, linestyles='--',
            )
        except (ValueError, TypeError):
            pass
        ax_p.set_xlabel('Max angle cutoff (<)', fontsize=9)
        ax_p.set_ylabel('Min angle cutoff (>)', fontsize=9)
        ax_p.set_title('Significance', fontsize=10, pad=4)
        ax_p.set_xticks(hi_ticks)
        ax_p.set_yticks(lo_ticks)
        ax_p.tick_params(labelsize=8)
        p_axes.append(ax_p)

    if d_im is not None:
        fig.colorbar(d_im, ax=d_axes, shrink=0.7, pad=0.02, label='Mean D (degrees)')
    if p_im is not None:
        cb = fig.colorbar(p_im, ax=p_axes, shrink=0.7, pad=0.02, label=r'$-\log_{10}(p)$')
        p05_line = -np.log10(0.05)
        # Only draw the p=.05 marker if it falls within the colorbar range
        if p05_line <= logp_vmax:
            cb.ax.axhline(p05_line, color='cyan', linewidth=1.5, linestyle='--')
            cb.ax.text(0.5, p05_line, 'p=.05', va='center', ha='center', fontsize=8,
                       color='cyan', transform=cb.ax.get_yaxis_transform())

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved {save}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


# ── User angle vs critical angle correlation ────────────────────────────────

# Geometric critical tipping angle per tube type: arctan(W/H)
_TUBE_CRITICAL = {
    0: np.degrees(np.arctan(24 / 130)),   # 10.5°  tall narrow
    1: np.degrees(np.arctan(48 / 130)),   # 20.3°  wide tall
    2: np.degrees(np.arctan(24 / 65)),    # 20.3°  narrow short  (same ratio as [1])
    3: np.degrees(np.arctan(48 / 65)),    # 36.4°  wide short
}
_TUBE_DIMS = {0: '24×130', 1: '48×130', 2: '24×65', 3: '48×65'}


def angle_correlation(
    sessions: "TubeSessions",
    *,
    save: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot per-user mean tilt angle vs tube critical angle (arctan W/H).

    Tubes [1] and [2] share the same critical angle (20.3°) and are averaged
    into a single x-position. Each user's three per-difficulty means are
    connected as a spaghetti line. A per-user slope histogram (right panel)
    shows how well individual participants track tube difficulty.

    save='path.png' — save to file.
    show=True       — display interactively.
    """
    import pandas as pd
    from scipy import stats as sp_stats

    all_trials = pd.concat(
        [sess.trials for sess in sessions], ignore_index=True
    )
    valid = all_trials[all_trials['valid'] == True].copy()

    if 'tube_type_index' not in valid.columns:
        print("  angle_correlation: tube_type_index column not found.")
        return None

    # Average tubes [1] and [2] (identical critical angle) → 3 x-positions
    valid['_tube_group'] = valid['tube_type_index'].map({0: 0, 1: 1, 2: 1, 3: 3})
    valid['_crit'] = valid['_tube_group'].map(
        {0: _TUBE_CRITICAL[0], 1: _TUBE_CRITICAL[1], 3: _TUBE_CRITICAL[3]}
    )

    per_user = (
        valid.groupby(['user_number', '_crit'])
             .agg(mean_angle=('angle', 'mean'))
             .reset_index()
    )
    # Keep only users with all 3 difficulty levels
    counts = per_user.groupby('user_number')['_crit'].count()
    per_user = per_user[per_user['user_number'].isin(counts[counts == 3].index)]
    n_users = per_user['user_number'].nunique()

    x_vals = sorted(per_user['_crit'].unique())

    group_stats = (
        per_user.groupby('_crit')['mean_angle']
                .agg(['mean', 'std', 'count'])
                .reset_index()
    )
    group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])

    user_slopes = [
        sp_stats.linregress(grp['_crit'], grp['mean_angle'])[0]
        for _, grp in per_user.groupby('user_number')
    ]
    slopes = np.array(user_slopes)

    r_all, p_all = sp_stats.pearsonr(per_user['_crit'], per_user['mean_angle'])
    sl_all, ic_all, *_ = sp_stats.linregress(per_user['_crit'], per_user['mean_angle'])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.32,
                           left=0.08, right=0.97, top=0.90, bottom=0.12)
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # Spaghetti
    for _, grp in per_user.groupby('user_number'):
        grp_s = grp.sort_values('_crit')
        ax_main.plot(grp_s['_crit'], grp_s['mean_angle'],
                     color='#4A90D9', alpha=0.06, linewidth=0.8, zorder=1)
        ax_main.scatter(grp_s['_crit'], grp_s['mean_angle'],
                        color='#4A90D9', alpha=0.08, s=6, zorder=2)

    # Group means ± 95 % CI
    ax_main.errorbar(group_stats['_crit'], group_stats['mean'],
                     yerr=group_stats['se'] * 1.96,
                     fmt='o', color='#1a1a2e', markersize=8,
                     capsize=5, linewidth=2.5, zorder=4, label='Mean ± 95% CI')

    # Regression
    x_line = np.linspace(8, 40, 100)
    ax_main.plot(x_line, sl_all * x_line + ic_all,
                 color='#e63946', linewidth=2, linestyle='--', zorder=3,
                 label=f'Regression (r={r_all:.2f}, p<.001)')

    # y = x reference
    ax_main.plot(x_line, x_line, color='#888', linewidth=1, linestyle=':',
                 alpha=0.6, zorder=2, label='y = x  (perfect calibration)')

    # Overshoot / undershoot shading
    ax_main.fill_between(x_line, x_line, 50, alpha=0.04, color='#2a9d8f', zorder=0)
    ax_main.fill_between(x_line, 0, x_line, alpha=0.04, color='#e63946', zorder=0)
    ax_main.text(33, 47, 'overshoot', fontsize=8, color='#2a9d8f', ha='center')
    ax_main.text(33, 28, 'undershoot', fontsize=8, color='#e63946', ha='center')

    # X-axis labels with tube info
    ax_main.set_xticks(x_vals)
    ax_main.set_xticklabels([
        f'Tube [0]\n24×130\n({x_vals[0]:.1f}°)',
        f'Tubes [1,2]\n48×130 / 24×65\n({x_vals[1]:.1f}°)',
        f'Tube [3]\n48×65\n({x_vals[2]:.1f}°)',
    ], fontsize=9)

    ax_main.set_xlabel('Critical tipping angle  arctan(W/H)', fontsize=11)
    ax_main.set_ylabel('User mean tilt angle (°)', fontsize=11)
    ax_main.set_title(
        f'User Angle vs. Tube Critical Angle  (N={n_users} users)',
        fontsize=12, fontweight='bold',
    )
    ax_main.legend(fontsize=9, loc='upper left')
    ax_main.set_xlim(7, 39)
    ax_main.set_ylim(0, 50)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Slope histogram
    ax_hist.hist(slopes, bins=30, color='#4A90D9', edgecolor='white',
                 linewidth=0.5, alpha=0.85)
    ax_hist.axvline(np.median(slopes), color='#1a1a2e', linewidth=2,
                    linestyle='--', label=f'Median={np.median(slopes):.2f}')
    ax_hist.axvline(0, color='#e63946', linewidth=1.5, linestyle=':')
    pct_pos = (slopes > 0).mean() * 100
    ax_hist.text(0.97, 0.97, f'{pct_pos:.0f}% slope > 0',
                 ha='right', va='top', transform=ax_hist.transAxes,
                 fontsize=9, color='#333')
    ax_hist.set_xlabel('Per-user slope\n(°user / °critical)', fontsize=10)
    ax_hist.set_ylabel('# users', fontsize=10)
    ax_hist.set_title('Sensitivity to\ntube difficulty', fontsize=11, fontweight='bold')
    ax_hist.legend(fontsize=8)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    if save:
        fig.savefig(save, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved {save}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig
