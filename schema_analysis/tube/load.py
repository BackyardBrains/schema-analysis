"""
Data loading for tube-tilt experiments. Self-contained — no other project imports.

All JSON-based experiments (Exp1, Exp2, future) use the identical pipeline:

    load_from_json(directory, face_id=None)   → trials DataFrame (unified schema)

faceType in the JSON determines how face columns are resolved:
  - 'ID015', 'ID030', etc. (matches ID\\d+) → face_id = faceType, eyes_covered = False
  - 'sighted', 'blindfold', etc.            → face_id from parameter, eyes_covered from faceType

Exp2 raw data lives in data/tube/raw/exp2/ (never modified).  Before analysis,
quarantine_workers() classifies files into:
  - exp2/quarantined/  (repeat-worker files — bots)
  - exp2/user-data/    (single-session worker files + files without workerId)

load_exp2() loads from user-data/ (or the raw directory if user-data/ doesn't
exist yet).  CSV fallback files are always loaded from the raw directory.

Exports:
    quarantine_workers(directory)      → classify raw files into quarantined/ + user-data/
    load_from_json(directory, face_id) → unified trials DataFrame (all JSON experiments)
    load_exp1()                        → Exp1 loader
    load_exp2()                        → Exp2 loader
    flag_bots(trials_df)              → set of bot UUIDs
    mark_valid(df, lo, hi)            → df with 'valid' column
    balance_cascade(df, condition_col) → df with updated 'valid' column
"""

import glob
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

ANGLE_LO = 3    # exclusive lower bound (degrees)
ANGLE_HI = 40   # exclusive upper bound (degrees)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EXP1_JSON_DIR = os.path.join(ROOT, 'data', 'tube', 'raw', 'exp1')
_EXP2_DIR      = os.path.join(ROOT, 'data', 'tube', 'raw', 'exp2')

_FACE_ID_PATTERN = re.compile(r'^ID\d+$')


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _parse_session(raw):
    uuid = raw['UUID']
    data = raw.get('data', {})
    session = data.get('session', {})
    trials = data.get('trials', [])
    return {
        'uuid': uuid,
        'session_group': session.get('session_group', ''),
        'experiment_version': session.get('experiment_version', ''),
        'n_trials': len(trials),
    }


def _parse_trials(raw):
    uuid = raw['UUID']
    data = raw.get('data', {})
    session = data.get('session', {})
    trials = data.get('trials', [])
    file_version = session.get('file_version', '')
    has_latency = file_version >= '1.4' if file_version else False
    worker_id = session.get('mturk', {}).get('workerId', '') if isinstance(session.get('mturk'), dict) else ''

    rows = []
    for t in trials:
        face_type = t.get('faceType', t.get('faceTyoe', t.get('facetype', '')))
        if not face_type:
            continue
        rows.append({
            'uuid': uuid,
            'workerId': worker_id,
            'trial_index': t.get('trialIndex', 0),
            'tube_type_index': t.get('tubeTypeIndex', 0),
            'arrow_direction': t.get('arrowDirection', ''),
            'face_type': face_type,
            'face_side': t.get('faceSide', ''),
            'end_angle': t.get('endAngle', 0),
            'latency': t.get('latency', np.nan) if has_latency else np.nan,
        })
    return rows


def _load_json_dir(directory):
    """Load all JSON files from a directory. Returns (sessions_df, trials_df)."""
    json_files = sorted(glob.glob(os.path.join(directory, '*.json')))
    if not json_files:
        print(f"  No JSON files found in {directory}")
        return pd.DataFrame(), pd.DataFrame()

    sessions, all_trials = [], []
    seen_uuids = set()
    skipped_copies = 0

    for filepath in json_files:
        basename = os.path.basename(filepath)
        name = os.path.splitext(basename)[0]
        if name != name.rstrip() or ' 2' in basename or ' 3' in basename:
            skipped_copies += 1
            continue
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARNING: Skipping malformed JSON: {basename}")
            continue

        uuid = raw.get('UUID', '')
        if not uuid or uuid in seen_uuids:
            continue
        seen_uuids.add(uuid)
        sessions.append(_parse_session(raw))
        all_trials.extend(_parse_trials(raw))

    sessions_df = pd.DataFrame(sessions)
    trials_df = pd.DataFrame(all_trials)
    print(f"  Loaded {len(seen_uuids)} unique participants from {len(json_files)} files")
    if skipped_copies:
        print(f"  Skipped {skipped_copies} copy artifacts")
    if not sessions_df.empty:
        print(f"  Versions: {sessions_df['experiment_version'].value_counts().to_dict()}")
    return sessions_df, trials_df


# ── Angle transform (applied once, at load time, for all JSON experiments) ────

def _transform_angles(df):
    """
    end_angle → raw_angle, tip_direction, angle.

    raw_angle : signed endAngle from JS (0=vertical, negative=left, positive=right)
    tip_direction : 'left' if raw_angle < 0 else 'right'
    angle : sign-flipped to tip direction — positive = tilted in arrow's direction,
            negative = tilted against it. No abs() — the sign is meaningful.
    """
    df = df.copy()
    df['raw_angle'] = df['end_angle']
    df['tip_direction'] = np.where(df['end_angle'] < 0, 'left', 'right')
    df['angle'] = np.where(df['tip_direction'] == 'left', -df['end_angle'], df['end_angle'])
    return df


def _derive_towards_away(df):
    """arrow_direction × face_side → towards/away."""
    df = df.copy()
    df['towards_away'] = np.where(
        df['arrow_direction'] == df['face_side'], 'towards', 'away'
    )
    return df


def _resolve_face_columns(df, face_id=None):
    """
    Resolve face_id and eyes_covered from face_type column.

    If face_type matches ID\\d+ (e.g. ID015, ID030):
        → face_id = face_type value, eyes_covered = False
    Otherwise (e.g. 'sighted', 'blindfold'):
        → face_id = provided parameter, eyes_covered = (face_type == 'blindfold')
    """
    df = df.copy()
    if 'face_type' not in df.columns:
        df['face_id'] = face_id or 'UNKNOWN'
        df['eyes_covered'] = False
        return df

    sample = df['face_type'].dropna().iloc[0] if not df['face_type'].dropna().empty else ''
    if _FACE_ID_PATTERN.match(str(sample)):
        # faceType IS the face identifier (threat experiment, etc.)
        df['face_id'] = df['face_type']
        df['eyes_covered'] = False
    else:
        # faceType is a condition label (sighted/blindfold experiment)
        df['face_id'] = face_id or 'UNKNOWN'
        df['eyes_covered'] = df['face_type'].str.lower() == 'blindfold'

    return df


def _unify_columns(df, sessions_df):
    """Add user_number, session_group, and rename to unified camelCase schema."""
    df = df.copy()

    # Sequential user_number from uuid
    uid_map = {u: i + 1 for i, u in enumerate(sorted(df['uuid'].unique()))}
    df['user_number'] = df['uuid'].map(uid_map)

    # session_group from sessions_df
    if not sessions_df.empty and 'session_group' in sessions_df.columns:
        sg_map = sessions_df.set_index('uuid')['session_group'].to_dict()
        df['session_group'] = df['uuid'].map(sg_map)
    else:
        df['session_group'] = ''

    # Canonical camelCase aliases expected by the rest of the pipeline
    df['tubeTypeIndex'] = df['tube_type_index']
    df['faceSide'] = df['face_side']
    df['trialIndex'] = df['trial_index']

    return df


# ── Validation ────────────────────────────────────────────────────────────────

def mark_valid(df, lo=ANGLE_LO, hi=ANGLE_HI):
    """Add/update 'valid' column: True when lo < angle < hi."""
    df = df.copy()
    df['valid'] = (df['angle'] > lo) & (df['angle'] < hi)
    return df


# ── Balance cascade ───────────────────────────────────────────────────────────

def balance_cascade(df, condition_col):
    """
    Single-pass invalidation: when a trial is invalid, invalidate its towards/away partner.

    Match = same (user, tube, tip_direction, condition) but opposite towards/away.
    Single pass only — iterating over the originally-invalid trials once. A while-loop
    would over-cascade: once A1 is invalidated it would hunt for ITS partner,
    potentially killing an unrelated valid trial.
    """
    df = df.copy()
    tube_col = 'tubeTypeIndex' if 'tubeTypeIndex' in df.columns else 'tube_type_index'
    user_col = 'user_number' if 'user_number' in df.columns else 'uuid'

    for uid in df[user_col].unique():
        ud = df[df[user_col] == uid].copy()
        for idx, inv in ud[ud['valid'] == False].iterrows():
            opp = 'away' if inv['towards_away'] == 'towards' else 'towards'
            match = ud[
                (ud['valid'] == True) &
                (ud[tube_col] == inv[tube_col]) &
                (ud['tip_direction'] == inv['tip_direction']) &
                (ud[condition_col] == inv[condition_col]) &
                (ud['towards_away'] == opp)
            ].index
            if not match.empty:
                ud.at[match[0], 'valid'] = False
        df.loc[df[user_col] == uid, 'valid'] = ud['valid'].values
    return df


# ── Bot detection ─────────────────────────────────────────────────────────────

_EXPECTED_TRIALS = 32


def flag_bots(trials_df):
    """
    Return set of bot UUIDs (HIGH + MEDIUM risk) based on behavioural flags.

    Flags: all_zero_angles, low_angle_variability, wrong_direction,
           fast_median_rt, low_rt_variability, mostly_fast_rts, very_fast_total.
    is_bot = True for >= 2 flags.
    """
    bot_uuids = set()
    for uuid, grp in trials_df.groupby('uuid'):
        angles = grp['end_angle'].values if 'end_angle' in grp.columns else grp.get('raw_angle', pd.Series()).values
        latencies = grp['latency'].values if 'latency' in grp.columns else np.full(len(grp), np.nan)
        n_trials = len(grp)

        has_latency = not np.all(np.isnan(latencies))
        mask_nf = (grp['trial_index'].values if 'trial_index' in grp.columns else
                   grp['trialIndex'].values if 'trialIndex' in grp.columns else
                   np.arange(n_trials)) > 0
        lat_nf = latencies[mask_nf]
        lat_nf = lat_nf[~np.isnan(lat_nf)] if has_latency else np.array([])

        pct_zero = np.sum(angles == 0) / len(angles) * 100 if len(angles) > 0 else 0
        sd_angle = np.std(angles) if len(angles) > 0 else 0

        n_flags = 0
        if pct_zero >= 90:
            n_flags += 1
        if sd_angle < 2 and pct_zero < 90:
            n_flags += 1

        direction_col = 'arrow_direction' if 'arrow_direction' in grp.columns else None
        if direction_col:
            dirs = grp[direction_col].values
            non_zero = angles != 0
            n_nz = non_zero.sum()
            if n_nz > 5:
                correct = sum(
                    (d == 'right' and a > 0) or (d == 'left' and a < 0)
                    for a, d in zip(angles, dirs) if a != 0
                )
                if correct / n_nz * 100 < 50:
                    n_flags += 1

        if has_latency and len(lat_nf) > 0:
            median_rt = np.median(lat_nf)
            mean_rt = np.mean(lat_nf)
            sd_rt = np.std(lat_nf)
            cv_rt = sd_rt / mean_rt if mean_rt > 0 else np.nan
            pct_under_2000 = np.sum(lat_nf < 2000) / len(lat_nf) * 100
            total_time_s = np.nansum(latencies[mask_nf]) / 1000

            if median_rt < 1000:
                n_flags += 1
            if not np.isnan(cv_rt) and cv_rt < 0.15 and n_trials == _EXPECTED_TRIALS:
                n_flags += 1
            if pct_under_2000 > 80:
                n_flags += 1
            if total_time_s < 60:
                n_flags += 1

        if n_flags >= 2:
            bot_uuids.add(uuid)

    return bot_uuids


# ── Worker quarantine ─────────────────────────────────────────────────────────

def quarantine_workers(directory, max_sessions=1):
    """
    Classify raw JSON files into quarantined/ and user-data/ subfolders.

    Reads every JSON in *directory*, groups by workerId.  Workers with
    more than *max_sessions* unique UUIDs have ALL their files copied to
    quarantined/.  Everything else (including files without a workerId)
    goes to user-data/.

    Both output folders are deleted and recreated from scratch each run,
    so raw/ is never modified.  A manifest.json is written to quarantined/
    for audit.

    Returns a dict summarising what happened.
    """
    quarantined_dir = os.path.join(directory, 'quarantined')
    userdata_dir = os.path.join(directory, 'user-data')

    # Wipe previous output so the classification is always fresh
    for d in (quarantined_dir, userdata_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Scan every top-level JSON and map workerId → list of (uuid, filepath)
    worker_files = defaultdict(list)
    no_worker_files = []

    json_files = sorted(glob.glob(os.path.join(directory, '*.json')))
    for filepath in json_files:
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        uuid = raw.get('UUID', '')
        mturk = raw.get('data', {}).get('session', {}).get('mturk', {})
        wid = mturk.get('workerId', '') if isinstance(mturk, dict) else ''
        if wid:
            worker_files[wid].append({'uuid': uuid, 'path': filepath})
        else:
            no_worker_files.append({'uuid': uuid, 'path': filepath})

    # Classify
    quarantined_workers = {}
    n_quarantined = 0
    n_userdata = 0

    for wid, entries in worker_files.items():
        unique_uuids = {e['uuid'] for e in entries}
        if len(unique_uuids) > max_sessions:
            quarantined_workers[wid] = {
                'sessions': len(unique_uuids),
                'files': len(entries),
                'uuids': sorted(unique_uuids),
            }
            for e in entries:
                shutil.copy2(e['path'], quarantined_dir)
                n_quarantined += 1
        else:
            for e in entries:
                shutil.copy2(e['path'], userdata_dir)
                n_userdata += 1

    # Files without workerId always go to user-data
    for e in no_worker_files:
        shutil.copy2(e['path'], userdata_dir)
        n_userdata += 1

    # Also copy CSV files to user-data (run1 fallback etc.)
    for csv_path in sorted(glob.glob(os.path.join(directory, '*.csv'))):
        shutil.copy2(csv_path, userdata_dir)

    # Write manifest
    manifest = {
        'generated': datetime.now().isoformat(),
        'source_directory': directory,
        'max_sessions': max_sessions,
        'total_json_files': len(json_files),
        'quarantined_files': n_quarantined,
        'quarantined_workers': len(quarantined_workers),
        'userdata_files': n_userdata,
        'no_worker_id_files': len(no_worker_files),
        'workers': quarantined_workers,
    }
    with open(os.path.join(quarantined_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  Quarantine: {n_quarantined} files from {len(quarantined_workers)} repeat workers → quarantined/")
    print(f"  User data:  {n_userdata} files ({len(no_worker_files)} without workerId) → user-data/")

    return manifest


# ── Generic JSON loader (all experiments) ─────────────────────────────────────

def load_from_json(directory, face_id=None):
    """
    Load any tube-tilt experiment from raw JSON files.

    Applies the identical pipeline for every JSON-based experiment:
        parse → transform angles → derive towards/away → resolve face columns → unify schema

    face_id parameter:
        Required when faceType in JSON is a condition label ('sighted'/'blindfold').
        Ignored when faceType is already a face identifier ('ID015', 'ID030', etc.).

    Returns unified trials DataFrame ready for validate_trials() → balance() → compute_d().
    """
    sessions_df, trials_df = _load_json_dir(directory)
    if trials_df.empty:
        return pd.DataFrame()

    trials_df = _transform_angles(trials_df)
    trials_df = _derive_towards_away(trials_df)
    trials_df = _resolve_face_columns(trials_df, face_id=face_id)
    trials_df = _unify_columns(trials_df, sessions_df)

    return trials_df


# ── Named loaders ─────────────────────────────────────────────────────────────

def load_exp1(json_dir=None):
    """Load Exp1 (ID008, sighted vs blindfold) from raw JSON."""
    return load_from_json(json_dir or _EXP1_JSON_DIR, face_id='ID008')


def load_exp2(exp2_dir=None):
    """
    Load Exp2 data.  If user-data/ subfolder exists (created by
    quarantine_workers), loads from there.  Otherwise falls back to the
    raw directory.  CSV fallback files are loaded from whichever
    directory contains them.
    """
    raw_dir = exp2_dir or _EXP2_DIR
    userdata_dir = os.path.join(raw_dir, 'user-data')
    directory = userdata_dir if os.path.isdir(userdata_dir) else raw_dir
    if directory == userdata_dir:
        print("  Loading Exp2 from user-data/ (post-quarantine)")
    parts = []

    # ── JSON files ────────────────────────────────────────────────────────────
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if json_files:
        df_json = load_from_json(directory)
        if not df_json.empty:
            parts.append(df_json)

    # ── CSV files (fallback for runs with lost JSONs) ─────────────────────────
    csv_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    for csv_path in csv_files:
        df_csv = pd.read_csv(csv_path)
        if 'eyes_covered' not in df_csv.columns:
            df_csv['eyes_covered'] = False
        print(f"  Loaded CSV: {os.path.basename(csv_path)} ({len(df_csv['uuid'].nunique() if 'uuid' in df_csv.columns else df_csv)} rows)")
        parts.append(df_csv)

    if not parts:
        print(f"  No Exp2 data found in {directory}")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)

    if 'uuid' in combined.columns:
        combined['_uid_key'] = combined['uuid'].where(
            combined['uuid'].notna() & (combined['uuid'] != ''),
            other='csv_user_' + combined['user_number'].astype(str),
        )
    else:
        combined['_uid_key'] = 'csv_user_' + combined['user_number'].astype(str)

    uid_map = {u: i + 1 for i, u in enumerate(sorted(combined['_uid_key'].unique()))}
    combined['user_number'] = combined['_uid_key'].map(uid_map)
    combined.drop(columns=['_uid_key'], inplace=True)

    n_total = combined['user_number'].nunique()
    print(f"  Exp2 combined: {n_total} unique participants")
    return combined
