"""
Main analysis script for the Star (Implied Motion) Experiment.

Loads the dataset, assigns users to their respective sub-experiments 
(Exp 1: Bars, Exp 2a: Eyes Open vs Closed, Exp 2b: Facing Towards vs Away),
calculates per-participant differences (Congruent - Incongruent),
runs statistical t-tests, and plots the delta effects.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as sp_stats

from schema_analysis.star.load import load_star_data

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
IMG_DIR = os.path.join(ROOT, 'schema_analysis', 'star', 'img')

def get_image(filename, target_zoom_height=350):
    """Load a stimulus image for embedding into a plot."""
    path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(path):
        return None
    img = plt.imread(path)
    h, w = img.shape[:2]
    target_zoom = 0.45 * (target_zoom_height / h)
    return OffsetImage(img, zoom=target_zoom)

def add_image_annotation(ax, image_box, x, y):
    """Add the image to the specific coordinates on the axes."""
    if image_box is None: return
    ab = AnnotationBbox(image_box, (x, y), frameon=False, 
                        xycoords=('data', 'axes fraction'), 
                        box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

def draw_vertical_divider(ax, x):
    """Draw a graphical dashed line between sub-experiments."""
    ln = ax.axvline(x=x, color='#bbbbbb', linewidth=1.5, linestyle='--', zorder=0)
    ln.set_clip_on(False)

def run_analysis():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading star experiment data...")
    sessions_df, trials_df = load_star_data()
    
    if trials_df.empty:
        print("No data found!")
        return

    # Filter out invalid trials and low accuracy users
    valid_trials = trials_df[(trials_df['valid'] == True) & (trials_df['high_accuracy'] == True)].copy()
    print(f"Total: {len(valid_trials)} valid trials from high-accuracy participants.")

    # -----------------------------------------------------
    # Assign and pool UUIDs by Sub-Experiment
    # -----------------------------------------------------
    exp1_uuids = set(valid_trials[valid_trials['condition'] == 'bars']['uuid'])
    exp2a_uuids = set(valid_trials[(valid_trials['condition'] == 'faces') & (valid_trials['eyesOpen'] == False)]['uuid'])
    exp2b_uuids = set(valid_trials[(valid_trials['condition'] == 'faces') & (valid_trials['gazeTowards'] == False)]['uuid'])

    print(f"Pool sizes: Exp 1 (N={len(exp1_uuids)}), Exp 2a (N={len(exp2a_uuids)}), Exp 2b (N={len(exp2b_uuids)})")

    t_exp1 = valid_trials[valid_trials['uuid'].isin(exp1_uuids)]
    t_exp2a = valid_trials[valid_trials['uuid'].isin(exp2a_uuids)]
    t_exp2b = valid_trials[valid_trials['uuid'].isin(exp2b_uuids)]

    # -----------------------------------------------------
    # Compute per-participant difference (Congruent - Incongruent)
    # -----------------------------------------------------
    def get_stats(df, subset_filter):
        subset = df[subset_filter]
        
        # We need per-uuid differences to run the paired t-test
        user_acc = subset.groupby(['uuid', 'congruent'])['correct'].mean() * 100
        user_rt = subset[subset['correct'] == True].groupby(['uuid', 'congruent'])['rt'].mean()

        def calc_diffs(ser):
            try:
                unstacked = ser.unstack(level='congruent')
                if True in unstacked.columns and False in unstacked.columns:
                    # Difference: Congruent - Incongruent
                    return (unstacked[True] - unstacked[False]).dropna()
            except Exception:
                pass
            return pd.Series(dtype=float)

        acc_diffs = calc_diffs(user_acc)
        rt_diffs = calc_diffs(user_rt)
        
        # Means and SEM
        acc_mean = acc_diffs.mean() if len(acc_diffs) else 0
        acc_sem = acc_diffs.sem() if len(acc_diffs) else 0
        
        rt_mean = rt_diffs.mean() if len(rt_diffs) else 0
        rt_sem = rt_diffs.sem() if len(rt_diffs) else 0
        
        # Statistical tests (1-sample t-test testing difference against 0)
        acc_p = sp_stats.ttest_1samp(acc_diffs, 0).pvalue if len(acc_diffs) > 1 else 1.0
        rt_p = sp_stats.ttest_1samp(rt_diffs, 0).pvalue if len(rt_diffs) > 1 else 1.0
        
        return {
            'acc': acc_mean, 'acc_sem': acc_sem, 'acc_p': acc_p,
            'rt': rt_mean, 'rt_sem': rt_sem, 'rt_p': rt_p,
            'N': len(rt_diffs)
        }

    # Data array aligning with the 5 X-axis slots
    stats = [
        get_stats(t_exp1, t_exp1['condition'] == 'bars'),
        get_stats(t_exp2a, t_exp2a['eyesOpen'] == True),
        get_stats(t_exp2a, t_exp2a['eyesOpen'] == False),
        get_stats(t_exp2b, t_exp2b['gazeTowards'] == True),
        get_stats(t_exp2b, t_exp2b['gazeTowards'] == False)
    ]

    labels = [
        'A\nBars (Control)',
        'B\nEyes Open\n(Twds)',
        'B\nBlindfolded\n(Twds)',
        'C\nFacing Towards\n(Open)',
        'C\nFacing Away\n(Open)'
    ]
    
    images = [
        'Grating.png', 
        'BlankFaceLookingRight (1).png', 
        'BlankFaceLookingRightBlindfold (1).png',
        'BlankFaceLookingRight (1).png', 
        'BlankFaceLookingLeft (1).png'
    ]

    def plot_metric(metric, title, ylabel, filename):
        fig = plt.figure(figsize=(15, 8.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[0.6, 1], hspace=0.1,
                               left=0.08, right=0.98, top=0.88, bottom=0.1)
        ax_img = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
        ax_img.set_axis_off()
        
        x = np.arange(len(stats))
        ax_img.set_xlim(-0.5, len(stats) - 0.5)
        width = 0.55
        
        # Render Images in Top Axes
        for i, img_name in enumerate(images):
            img_box = get_image(img_name)
            if img_box:
                add_image_annotation(ax_img, img_box, x[i], 0.35)

        vals = [s[metric] for s in stats]
        errs = [s[f'{metric}_sem'] for s in stats]
        pvals = [s[f'{metric}_p'] for s in stats]
        n_counts = [s['N'] for s in stats]
        
        # Use single color for bars since they all represent the difference
        colors = ['#888888'] * len(stats)
        # Highlight significant ones
        colors[0] = '#4C72B0' if pvals[0] < 0.05 else '#888888'
        
        rects = ax_bar.bar(x, vals, width, yerr=errs, capsize=5, color='#4A90D9', edgecolor='black', linewidth=1)
        
        # Add N and significance markers
        yvals_ext = [abs(v) + e for v, e in zip(vals, errs)]
        ylim = max(yvals_ext) * 1.5 if max(yvals_ext) > 0 else 10

        for i, (val, err, pval, n_count) in enumerate(zip(vals, errs, pvals, n_counts)):
            # Render significance star
            if pval < 0.05:
                # Place star above positive bar, or below negative bar
                y_star = val + err + (max(max(vals), abs(min(vals))) * 0.05) if val >= 0 else val - err - (max(max(vals), abs(min(vals))) * 0.1)
                va_star = 'bottom' if val >= 0 else 'top'
                ax_bar.text(i, y_star, '*', ha='center', va=va_star, fontsize=24, fontweight='bold')
                
            y_pval = -ylim * 0.75
            ax_bar.text(i, y_pval, f"p={pval:.3f}", ha='center', va='center', fontsize=9, color='#444')
            
            # Print N exactly under the top of the bar figure
            ax_bar.text(i, 0.98, f'N={n_count}', ha='center', va='top', fontsize=10, transform=ax_bar.get_xaxis_transform())

        # Zero line
        ax_bar.axhline(0, color='black', linewidth=1)

        ax_bar.set_ylabel(ylabel, fontsize=12)
        fig.suptitle(title, fontsize=15, fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=11, fontweight='bold')
        
        ax_bar.set_ylim(-ylim, ylim)
        
        # Loose the walls
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        # Draw grouping dividers
        draw_vertical_divider(ax_bar, 0.5)
        draw_vertical_divider(ax_bar, 2.5)
        
        # Draw Group Texts on Top Axes
        transform = ax_img.get_xaxis_transform()
        ax_img.text(0, 0.85, 'Exp 1\n(Bars)', ha='center', va='bottom', fontsize=12, fontweight='bold', transform=transform)
        ax_img.text(1.5, 0.85, 'Exp 2a\n(Eyes Open vs Blindfolded)', ha='center', va='bottom', fontsize=12, fontweight='bold', transform=transform)
        ax_img.text(3.5, 0.85, 'Exp 2b\n(Facing Tree vs Facing Away)', ha='center', va='bottom', fontsize=12, fontweight='bold', transform=transform)

        plot_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {plot_path}")
        
        return stats

    print("\n--- Generating plots for Congruent - Incongruent Differences ---")
    st = plot_metric('acc', 'Accuracy Difference (Congruent - Incongruent)\n* p < 0.05 (1-sample t-test vs 0)', 'Δ Accuracy (%)', 'star_accuracy.png')
    st_rt = plot_metric('rt', 'Reaction Time Difference (Congruent - Incongruent)\n* p < 0.05 (1-sample t-test vs 0)', 'Δ Reaction Time (ms)', 'star_rt.png')

if __name__ == '__main__':
    run_analysis()
