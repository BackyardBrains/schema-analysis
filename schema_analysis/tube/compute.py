"""
D computation and sensitivity analysis for tube-tilt experiments.

    compute_d(sessions, split_by, value)   → {D, SE, p, N, raw}
    sweep_sensitivity(sessions, ...)       → sensitivity grid

sweep_sensitivity uses the same method as the main analysis:
  single-pass cascade for each (lo, hi) cutoff pair.

All inner loops use numpy (np.bincount for per-user D), so the full
13 × 26 sweep completes in seconds even for 700+ participants.
Requires sessions to still have all trials — call before select_trials().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from .sessions import TubeSessions, Session

LO_RANGE = np.arange(0, 11)   # min angle cutoff: 0 → 10 (trial kept if angle > lo)
HI_RANGE = np.arange(35, 56)  # max angle cutoff: 35 → 55 (trial kept if angle < hi)
MIN_N = 10


# ── D computation ──────────────────────────────────────────────────────────────

def compute_d(
    sessions: "TubeSessions",
    split_by: str,
    value: Any,
) -> Dict[str, Any]:
    """
    Compute D = mean(towards angle) − mean(away angle) across sessions.

    For each session: filter trials where split_by==value and valid==True,
    compute per-session D. Then aggregate: mean, SE, one-sample t-test vs 0.

    Returns: {D, SE, p, N, raw}
    """
    ds = []
    for sess in sessions:
        df = sess.trials
        if split_by not in df.columns:
            continue
        sub = df[(df[split_by] == value) & (df['valid'] == True)]
        if sub.empty:
            continue
        tw = sub.loc[sub['towards_away'] == 'towards', 'angle']
        aw = sub.loc[sub['towards_away'] == 'away', 'angle']
        if len(tw) > 0 and len(aw) > 0:
            ds.append(float(tw.mean() - aw.mean()))

    ds = np.array(ds)
    if len(ds) == 0:
        return dict(D=np.nan, SE=np.nan, p=np.nan, N=0, raw=ds)
    p = float(stats.ttest_1samp(ds, 0).pvalue) if len(ds) > 1 else np.nan
    return dict(
        D=float(np.mean(ds)),
        SE=float(np.std(ds) / np.sqrt(len(ds))) if len(ds) > 1 else np.nan,
        p=p,
        N=len(ds),
        raw=ds,
    )


# ── Partner pre-computation (vectorized) ───────────────────────────────────────

def _precompute_partners(df: pd.DataFrame, condition_col: str) -> np.ndarray:
    """
    Pre-compute the partner row index for each trial using vectorized groupby.

    Partner = same (user, tube, tip_direction, condition) but opposite
    towards/away, matched by position within the group (pair rank).
    Returns array of partner row positions relative to df (−1 if no partner).
    """
    tube_col = 'tubeTypeIndex' if 'tubeTypeIndex' in df.columns else 'tube_type_index'
    user_col = 'user_number' if 'user_number' in df.columns else 'uuid'

    key_cols = [c for c in [user_col, tube_col, 'tip_direction', condition_col]
                if c in df.columns]

    df = df.reset_index(drop=True)
    df['_pos'] = np.arange(len(df))
    df['_pair_rank'] = df.groupby(key_cols + ['towards_away']).cumcount()

    towards = df[df['towards_away'] == 'towards'][key_cols + ['_pair_rank', '_pos']]
    away    = df[df['towards_away'] == 'away'   ][key_cols + ['_pair_rank', '_pos']]

    merged = towards.merge(away, on=key_cols + ['_pair_rank'], suffixes=('_t', '_a'))

    partners = np.full(len(df), -1, dtype=np.int64)
    if not merged.empty:
        t_pos = merged['_pos_t'].values
        a_pos = merged['_pos_a'].values
        partners[t_pos] = a_pos
        partners[a_pos] = t_pos

    return partners


# ── Sensitivity sweep ──────────────────────────────────────────────────────────

def sweep_sensitivity(
    sessions: "TubeSessions",
    lo_range: Optional[np.ndarray] = None,
    hi_range: Optional[np.ndarray] = None,
    min_n: int = MIN_N,
) -> dict:
    """
    Sweep angle-acceptance cutoffs and compute D, p, N for each condition group.

    Uses the same method as the main analysis: for each (lo, hi) cutoff pair,
    re-marks valid by angle (absolute tip angle), applies single-pass cascade,
    then computes D = mean(towards) − mean(away) per participant.

    Requires sessions to still have ALL trials (valid and invalid) —
    call before select_trials().

    Performance: O(n_trials) per cutoff via np.bincount — no Python user loops.
    """
    lo_range = lo_range if lo_range is not None else LO_RANGE
    hi_range = hi_range if hi_range is not None else HI_RANGE

    # Collect ALL trials (including invalid) with experiment metadata
    all_dfs = []
    for sess in sessions:
        df = sess.trials.copy()
        df['_exp_num'] = sess.exp_num
        all_dfs.append(df)

    if not all_dfs:
        return dict(groups={}, group_labels={}, lo_range=lo_range,
                    hi_range=hi_range, min_n=min_n, sessions=sessions)

    combined = pd.concat(all_dfs, ignore_index=True).reset_index(drop=True)
    user_col = 'user_number' if 'user_number' in combined.columns else 'uuid'

    # Pre-compute partners per experiment (vectorized groupby, done once)
    partner_arr = np.full(len(combined), -1, dtype=np.int64)
    for exp_num, cond_col in [(1, 'eyes_covered'), (2, 'face_id')]:
        mask = (combined['_exp_num'] == exp_num).values
        if not mask.any() or cond_col not in combined.columns:
            continue
        sub = combined[mask].copy()
        sub_partners = _precompute_partners(sub, cond_col)
        # Map sub-positions → combined positions using numpy indexing
        positions = np.where(mask)[0]
        has_partner = sub_partners >= 0
        sub_with = np.where(has_partner)[0]
        partner_arr[positions[sub_with]] = positions[sub_partners[sub_with]]

    # Extract numpy arrays for zero-overhead inner loop
    angles       = combined['angle'].values.astype(float)
    user_ids_raw = combined[user_col].values
    face_ids     = combined['face_id'].values if 'face_id' in combined.columns else np.full(len(combined), '')
    eyes_covered = combined['eyes_covered'].values.astype(bool) if 'eyes_covered' in combined.columns else np.zeros(len(combined), dtype=bool)
    towards_away = combined['towards_away'].values

    # Integer-encode users for np.bincount (O(1) per-user aggregation)
    unique_users, user_int = np.unique(user_ids_raw, return_inverse=True)
    n_users = len(unique_users)

    is_towards = (towards_away == 'towards')
    is_away    = (towards_away == 'away')

    # Discover condition groups from data
    group_keys: dict = {}
    if 'face_id' in combined.columns and 'eyes_covered' in combined.columns:
        for _, row in combined[['face_id', 'eyes_covered']].drop_duplicates().iterrows():
            fid, ec = row['face_id'], bool(row['eyes_covered'])
            key = (fid, ec)
            from .treatments import resolve
            t = resolve(fid, ec)
            group_keys[key] = t['label']

    nL, nH = len(lo_range), len(hi_range)
    results = {key: {'D': np.full((nL, nH), np.nan),
                     'P': np.full((nL, nH), np.nan),
                     'N': np.full((nL, nH), np.nan)}
               for key in group_keys}

    # Pre-build angle-independent group masks
    group_base_masks = {
        key: (face_ids == key[0]) & (eyes_covered == key[1])
        for key in group_keys
    }

    print("  Sweeping cutoffs...", end="", flush=True)
    for i, lo in enumerate(lo_range):
        for j, hi in enumerate(hi_range):
            # Step 1: mark valid by absolute angle cutoff
            valid = (angles > lo) & (angles < hi)

            # Step 2: single-pass cascade using pre-computed partner array
            invalid_idx = np.where(~valid)[0]
            for idx in invalid_idx:
                p = partner_arr[idx]
                if p >= 0 and valid[p]:
                    valid[p] = False

            # Step 3: per-group D via np.bincount (no Python user loop)
            for key, base_mask in group_base_masks.items():
                tw_mask = valid & base_mask & is_towards
                aw_mask = valid & base_mask & is_away

                tw_sum = np.bincount(user_int[tw_mask], weights=angles[tw_mask], minlength=n_users)
                tw_cnt = np.bincount(user_int[tw_mask], minlength=n_users).astype(float)
                aw_sum = np.bincount(user_int[aw_mask], weights=angles[aw_mask], minlength=n_users)
                aw_cnt = np.bincount(user_int[aw_mask], minlength=n_users).astype(float)

                has_both = (tw_cnt > 0) & (aw_cnt > 0)
                n = has_both.sum()
                if n < 2:
                    continue

                ds = tw_sum[has_both] / tw_cnt[has_both] - aw_sum[has_both] / aw_cnt[has_both]
                _, p_val = stats.ttest_1samp(ds, 0)
                results[key]['D'][i, j] = float(ds.mean())
                results[key]['P'][i, j] = float(p_val)
                results[key]['N'][i, j] = int(n)

        print(".", end="", flush=True)
    print(" done.")

    return dict(
        groups=results,
        group_labels=group_keys,
        lo_range=lo_range,
        hi_range=hi_range,
        min_n=min_n,
        sessions=sessions,
    )
