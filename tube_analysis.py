#!/usr/bin/env python3
"""
Tube-tilt analysis: quarantine → load → clean → balance → print results → save plots.

Run:
    python tube_analysis.py

Output:
    Console: D results per condition (paste into LLMs)
    figure_d_bars.png          — D bar chart (Exp1 + Exp2)
    sensitivity_exp1.png       — angle cutoff sensitivity for Exp1
    sensitivity_exp2.png       — angle cutoff sensitivity for Exp2

--- SELECTION CRITERIA (edit here) -------------------------------------------
ANGLE_LO : trial kept if angle >  ANGLE_LO  (exclusive lower bound, degrees)
ANGLE_HI : trial kept if angle <  ANGLE_HI  (exclusive upper bound, degrees)
BOT_CLEAN: True  → remove bot-flagged participants before analysis
           False → keep all participants
"""

import os

from schema_analysis.tube import load
from schema_analysis.tube.load import quarantine_workers, _EXP2_DIR
from schema_analysis.tube.plots import sensitivity_heatmap

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Selection criteria ────────────────────────────────────────────────────────
ANGLE_LO  = 3      # degrees — lower cutoff (exclusive)
ANGLE_HI  = 40     # degrees — upper cutoff (exclusive)
BOT_CLEAN = True   # remove bot-flagged participants
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Tier 1: Quarantine repeat workers (file-level) ────────────────────────
    print("=" * 60)
    print("TIER 1 — Worker quarantine (repeat submissions)")
    print("=" * 60)
    manifest = quarantine_workers(_EXP2_DIR, max_sessions=1)
    print(f"  Total raw files:       {manifest['total_json_files']}")
    print(f"  Quarantined:           {manifest['quarantined_files']} files "
          f"({manifest['quarantined_workers']} workers with >1 session)")
    print(f"  User data:             {manifest['userdata_files']} files "
          f"({manifest['no_worker_id_files']} without workerId)")

    # ── Load from user-data/ ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    print(f"  [cutoffs: {ANGLE_LO}° < angle < {ANGLE_HI}°, "
          f"bots={'removed' if BOT_CLEAN else 'kept'}]")

    s = load(clean=False)
    s = s.exclude_face('ID030')

    # ── Tier 2: Behavioral bot detection ──────────────────────────────────────
    if BOT_CLEAN:
        s = s.remove_bad_sessions()

    s = s.validate_trials(min=ANGLE_LO, max=ANGLE_HI)
    s = s.balance()

    # Sensitivity sweeps (before select_trials)
    e1 = s.select(exp_num=1)
    if len(e1) > 0:
        out_s1 = os.path.join(ROOT, 'sensitivity_exp1.png')
        sensitivity_heatmap(
            e1,
            save=out_s1,
            title='Sensitivity Analysis — Exp 1 (ID008: Sighted vs Blindfold)',
        )

    e2 = s.select(exp_num=2)
    if len(e2) > 0:
        out_s2 = os.path.join(ROOT, 'sensitivity_exp2.png')
        sensitivity_heatmap(
            e2,
            save=out_s2,
            title='Sensitivity Analysis — Exp 2 (Threat Level: ID015 / ID017)',
        )

    # Main analysis — valid trials only, ID015 + ID017 paired
    s = s.select_trials('valid == True')

    print(f"\n{repr(s)}")
    s.print_summary()

    out_d = os.path.join(ROOT, 'figure_d_bars.png')
    s.plots.plot_d(save=out_d)

    print("\nDone.")


if __name__ == "__main__":
    main()
