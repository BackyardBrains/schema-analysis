"""
Session collection and pipeline for tube-tilt experiments.

    from schema_analysis.tube import load

    s = load(clean=True).balance()
    len(s)                          # sessions
    s[0]                            # one Session
    s.select(exp_num=1)             # Exp1 only
    s.select_trials('valid == True')  # filter trials in each session
    s.plots.plot_d(save='figure_d_bars.png')
    s.print_summary()
"""

import os
from typing import Any, Iterator, List, Optional

import numpy as np
import pandas as pd

from .load import (
    ANGLE_LO, ANGLE_HI,
    load_exp1, load_exp2,
    flag_bots, mark_valid, balance_cascade,
)
from .treatments import resolve as resolve_treatment

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Session ────────────────────────────────────────────────────────────────────

class Session:
    """
    One participant's trials. Treatment metadata (label, color, image_path, etc.)
    resolved from the treatments registry at build time.
    """

    def __init__(
        self,
        session_id,
        trials: pd.DataFrame,
        *,
        group: str = "",
        exp_num: int = 2,
    ):
        self.session_id = session_id
        self._trials = trials.copy()
        self.group = group
        self.exp_num = exp_num

        # Treatment properties — resolved once, read directly by plots/select
        self.face_id: Optional[str] = None
        self.eyes_covered: Optional[bool] = None
        self.label: Optional[str] = None
        self.color: Optional[str] = None
        self.image_path: Optional[str] = None
        self.gender: Optional[str] = None
        self.sight_type: Optional[str] = None
        self.threat_level: Optional[str] = None
        self._resolve_treatment()

    def _resolve_treatment(self) -> None:
        if self._trials.empty:
            return
        face_id = self._trials['face_id'].iloc[0] if 'face_id' in self._trials.columns else None
        if face_id is None:
            return
        ec_col = self._trials['eyes_covered'] if 'eyes_covered' in self._trials.columns else None
        eyes_covered = bool(ec_col.iloc[0]) if ec_col is not None else False
        t = resolve_treatment(face_id, eyes_covered)
        self.face_id = face_id
        self.eyes_covered = eyes_covered
        self.label = t['label']
        self.color = t['color']
        self.image_path = t['image_path']
        self.gender = t['gender']
        self.sight_type = t['sight_type']
        self.threat_level = t['threat_level']

    @property
    def trials(self) -> pd.DataFrame:
        return self._trials

    def select_trials(self, expr: str) -> "Session":
        """Return new Session with trials filtered by pandas query expression."""
        filtered = self._trials.query(expr)
        return Session(
            self.session_id, filtered,
            group=self.group, exp_num=self.exp_num,
        )

    def balance(self, condition_col: str) -> "Session":
        """Cascade invalidation: mark orphaned towards/away partners invalid."""
        self._trials = balance_cascade(self._trials, condition_col)
        return self

    def __len__(self) -> int:
        return len(self._trials)

    def __repr__(self) -> str:
        return (
            f"<Session {self.session_id}: {len(self)} trials, "
            f"group={self.group!r}, exp_num={self.exp_num}>"
        )


# ── TubeSessions ───────────────────────────────────────────────────────────────

class TubeSessions:
    """
    Collection of Session objects with a chainable pipeline.

    Pipeline phase  (DataFrames): remove_bad_sessions() → validate_trials() → balance()
    Collection phase (Sessions):  select() → select_trials() → plots → print_summary()
    """

    def __init__(self, sessions: Optional[List[Session]] = None):
        self._sessions: List[Session] = sessions or []
        self._built: bool = bool(sessions)
        # DataFrame state (pre-build)
        self._exp1_df: Optional[pd.DataFrame] = None
        self._exp2_df: Optional[pd.DataFrame] = None

    @classmethod
    def _from_dataframes(cls, exp1_df, exp2_df) -> "TubeSessions":
        obj = cls()
        obj._exp1_df = exp1_df
        obj._exp2_df = exp2_df
        obj._built = False
        return obj

    @classmethod
    def _from_sessions(cls, sessions: List[Session]) -> "TubeSessions":
        obj = cls(sessions=sessions)
        obj._built = True
        return obj

    # ── Pipeline (pre-balance) ─────────────────────────────────────────────────

    def remove_bad_sessions(self) -> "TubeSessions":
        """Remove participants flagged as bots. Only works for Exp1 (has uuid column)."""
        if self._exp1_df is None or 'uuid' not in self._exp1_df.columns:
            return self
        obj = self._copy_pipeline()
        bots = flag_bots(obj._exp1_df)
        if bots:
            obj._exp1_df = obj._exp1_df[~obj._exp1_df['uuid'].isin(bots)].copy()
            uid_map = {u: i + 1 for i, u in enumerate(sorted(obj._exp1_df['uuid'].unique()))}
            obj._exp1_df['user_number'] = obj._exp1_df['uuid'].map(uid_map)
        return obj

    def validate_trials(self, min: float = ANGLE_LO, max: float = ANGLE_HI) -> "TubeSessions":
        """Mark valid = (angle > min) & (angle < max) for all trial data."""
        obj = self._copy_pipeline()
        for attr in ('_exp1_df', '_exp2_df'):
            df = getattr(obj, attr)
            if df is not None and 'angle' in df.columns:
                setattr(obj, attr, mark_valid(df, lo=min, hi=max))
        return obj

    def balance(self) -> "TubeSessions":
        """Build sessions and cascade-invalidate orphaned trial pairs."""
        self._build()
        for sess in self._sessions:
            cond = 'eyes_covered' if sess.exp_num == 1 else 'face_id'
            sess.balance(cond)
        return self

    def _copy_pipeline(self) -> "TubeSessions":
        obj = TubeSessions()
        obj._exp1_df = self._exp1_df.copy() if self._exp1_df is not None else None
        obj._exp2_df = self._exp2_df.copy() if self._exp2_df is not None else None
        obj._built = False
        return obj

    def _build(self) -> None:
        """Build Session objects from DataFrames."""
        if self._built:
            return
        sessions: List[Session] = []
        for df, exp_num in [
            (self._exp1_df, 1),
            (self._exp2_df, 2),
        ]:
            if df is None or df.empty:
                continue
            if 'valid' not in df.columns:
                df = mark_valid(df, lo=ANGLE_LO, hi=ANGLE_HI)
            user_col = 'user_number' if 'user_number' in df.columns else 'uuid'
            for sid, grp in df.groupby(user_col):
                group = str(grp['session_group'].iloc[0]) if 'session_group' in grp.columns else ""
                sess = Session(sid, grp, group=group, exp_num=exp_num)
                sessions.append(sess)
        self._sessions = sessions
        self._built = True

    # ── Collection API ─────────────────────────────────────────────────────────

    def select(self, **kwargs: Any) -> "TubeSessions":
        """
        Filter sessions by attribute or trial-level column value.

        s.select(exp_num=1)             # Exp1 sessions
        s.select(threat_level='high')   # high-threat sessions
        s.select(face_id='ID015')       # sessions with ID015 trials
        """
        self._build()
        kept = []
        for sess in self._sessions:
            match = True
            for k, v in kwargs.items():
                attr_val = getattr(sess, k, None)
                if attr_val is not None and attr_val == v:
                    continue
                if k in sess.trials.columns and (sess.trials[k] == v).any():
                    continue
                match = False
                break
            if match:
                kept.append(sess)
        return TubeSessions._from_sessions(kept)

    def select_trials(self, expr: str) -> "TubeSessions":
        """
        Filter trials within each session using a pandas query expression.
        Returns new TubeSessions with only non-empty sessions.

        s.select_trials('valid == True')
        s.select_trials('angle > 5')
        """
        self._build()
        filtered = [sess.select_trials(expr) for sess in self._sessions if len(sess) > 0]
        filtered = [s for s in filtered if len(s) > 0]
        return TubeSessions._from_sessions(filtered)

    def __len__(self) -> int:
        self._build()
        return len(self._sessions)

    def __getitem__(self, idx: int) -> Session:
        self._build()
        return self._sessions[idx]

    def __iter__(self) -> Iterator[Session]:
        self._build()
        return iter(self._sessions)

    def __repr__(self) -> str:
        self._build()
        n_trials = sum(len(s) for s in self._sessions)
        return f"<TubeSessions: {len(self._sessions)} sessions, {n_trials} trials>"

    # ── Analysis ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print D analysis results to console (for pasting into LLMs)."""
        from .compute import compute_d

        self._build()
        print()
        print("=" * 60)
        print("TUBE-TILT ANALYSIS SUMMARY")
        print("=" * 60)

        # Exp1
        e1 = self.select(exp_num=1)
        if len(e1) > 0:
            print(f"\nExp 1  —  ID008 (N={len(e1)} sessions)")
            for split_val, label in [(False, 'Eyes Open'), (True, 'Eyes Covered')]:
                r = compute_d(e1, 'eyes_covered', split_val)
                if not np.isnan(r['D']):
                    sig = '*' if r['p'] < 0.05 else ''
                    print(f"  {label:20s}  N={r['N']:3d}   D={r['D']:+.3f}°   SE={r['SE']:.3f}   p={r['p']:.4f}  {sig}")

        # Exp2
        e2 = self.select(exp_num=2)
        if len(e2) > 0:
            print(f"\nExp 2  —  Threat faces (N={len(e2)} sessions)")
            for fid in ['ID015', 'ID017', 'ID030']:
                e2_face = e2.select(face_id=fid)
                if len(e2_face) == 0:
                    continue
                r = compute_d(e2, 'face_id', fid)
                if not np.isnan(r['D']):
                    sig = '*' if r['p'] < 0.05 else ''
                    label = f"{fid}"
                    print(f"  {label:20s}  N={r['N']:3d}   D={r['D']:+.3f}°   SE={r['SE']:.3f}   p={r['p']:.4f}  {sig}")

        print()

    # ── Plots ──────────────────────────────────────────────────────────────────

    @property
    def plots(self):
        """D figure object. Auto-discovers groups from sessions present."""
        from .plots import DFigure
        self._build()
        return DFigure(self)


# ── Entry point ────────────────────────────────────────────────────────────────

def load(
    *,
    clean: bool = False,
    exp1_json_dir: Optional[str] = None,
    exp2_dir: Optional[str] = None,
) -> TubeSessions:
    """
    Load all tube-tilt data. Returns TubeSessions for chaining.

    clean=True: remove bad sessions (Exp1 only) + validate trials with
                global defaults (ANGLE_LO=3, ANGLE_HI=40).

    exp2_dir: override the Exp2 data directory (default: data/tube/raw/exp2/).
              Any JSONs and CSVs found there are loaded and combined.

    Typical usage:
        s = load(clean=True).balance()
        s = s.select_trials('valid == True')
    """
    exp1_df = load_exp1(json_dir=exp1_json_dir)
    exp2_df = load_exp2(exp2_dir=exp2_dir)

    s = TubeSessions._from_dataframes(exp1_df, exp2_df)

    if clean:
        s = s.remove_bad_sessions().validate_trials(min=ANGLE_LO, max=ANGLE_HI)

    return s
