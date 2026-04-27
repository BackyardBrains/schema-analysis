"""
Data loading for star experiments (Implied Motion / RDK).

Parses both `rdk-bars` (Exp1 / adaptation control) and `rdk-face-1` (Exp2 / social attention).

Raw data lives in data/star/raw/ (never modified).  Before analysis,
quarantine_workers() classifies files into:
  - raw/quarantined/  (repeat-worker files — bots / duplicate submissions)
  - raw/user-data/    (single-session worker files + files without worker ID)

load_star_data() loads from user-data/ (or raw/ if user-data/ doesn't exist yet).
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

_PROLIFIC_PID_PATTERN = re.compile(r'^[0-9a-f]{24}$')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STAR_DIR = os.path.join(ROOT, 'data', 'star', 'raw')


def quarantine_workers(directory, max_sessions=1):
    """
    Classify raw JSON files into quarantined/ and user-data/ subfolders.

    Reads every JSON in *directory*, groups by workerId and prolificPid.
    Workers with more than *max_sessions* unique UUIDs have ALL their
    files copied to quarantined/.  Everything else (including files
    without an ID) goes to user-data/.

    Both output folders are deleted and recreated from scratch each run,
    so raw/ is never modified.  A manifest.json is written to quarantined/
    for audit.

    Returns a dict summarising what happened.
    """
    if not os.path.exists(directory):
        return

    quarantined_dir = os.path.join(directory, 'quarantined')
    userdata_dir = os.path.join(directory, 'user-data')

    for d in (quarantined_dir, userdata_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    json_files = sorted(glob.glob(os.path.join(directory, '*.json')))

    # Scan every top-level JSON and map identity → list of (uuid, filepath)
    worker_files = defaultdict(list)   # mturk workerId
    prolific_files = defaultdict(list) # prolific PID
    no_id_files = []
    fake_id_files = []

    for filepath in json_files:
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        uuid = raw.get('UUID', '')
        session = raw.get('data', {}).get('session', {})

        wid = ''
        mturk = session.get('mturk', {})
        if isinstance(mturk, dict):
            wid = mturk.get('workerId', '').strip()
            if wid and 'test' in wid.lower():
                wid = ''

        pid = ''
        prolific = session.get('prolific', {})
        if isinstance(prolific, dict):
            pid = prolific.get('prolificPid', '').strip()
            if pid and 'test' in pid.lower():
                pid = ''
            elif pid and not _PROLIFIC_PID_PATTERN.match(pid):
                fake_id_files.append({'uuid': uuid, 'path': filepath, 'fake_pid': pid})
                pid = ''

        entry = {'uuid': uuid, 'path': filepath}
        if wid:
            worker_files[wid].append(entry)
        if pid:
            prolific_files[pid].append(entry)
        if not wid and not pid and filepath not in {f['path'] for f in fake_id_files}:
            no_id_files.append(entry)

    # Build set of filepaths that should be quarantined
    quarantined_paths = set()
    quarantined_workers_detail = {}

    for wid, entries in worker_files.items():
        unique_uuids = {e['uuid'] for e in entries}
        if len(unique_uuids) > max_sessions:
            quarantined_workers_detail[f'mturk:{wid}'] = {
                'sessions': len(unique_uuids),
                'files': len(entries),
                'uuids': sorted(unique_uuids),
            }
            for e in entries:
                quarantined_paths.add(e['path'])

    for pid, entries in prolific_files.items():
        unique_uuids = {e['uuid'] for e in entries}
        if len(unique_uuids) > max_sessions:
            quarantined_workers_detail[f'prolific:{pid}'] = {
                'sessions': len(unique_uuids),
                'files': len(entries),
                'uuids': sorted(unique_uuids),
            }
            for e in entries:
                quarantined_paths.add(e['path'])

    # Quarantine fake Prolific IDs (test sessions)
    for entry in fake_id_files:
        quarantined_paths.add(entry['path'])
    if fake_id_files:
        quarantined_workers_detail['fake_prolific_ids'] = {
            'reason': 'prolificPid present but not a valid 24-char hex ID',
            'sessions': len(fake_id_files),
            'files': len(fake_id_files),
            'ids': sorted(set(e['fake_pid'] for e in fake_id_files)),
            'uuids': sorted(e['uuid'] for e in fake_id_files),
        }

    # Quarantine test sessions: face experiments with no platform ID,
    # and early prototype files (rdk_faces).
    test_sessions = []
    for filepath in json_files:
        if filepath in quarantined_paths:
            continue
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        exp = raw.get('experiment', '')
        uuid = raw.get('UUID', '')
        session = raw.get('data', {}).get('session', {})

        is_prototype = exp == 'rdk_faces'

        has_platform = False
        mturk = session.get('mturk', {})
        if isinstance(mturk, dict) and mturk.get('workerId', '').strip():
            has_platform = True
        prolific = session.get('prolific', {})
        if isinstance(prolific, dict) and _PROLIFIC_PID_PATTERN.match(
                prolific.get('prolificPid', '').strip()):
            has_platform = True

        is_face_no_id = 'face' in exp.lower() and not has_platform

        if is_prototype or is_face_no_id:
            quarantined_paths.add(filepath)
            test_sessions.append({'uuid': uuid, 'path': filepath,
                                  'experiment': exp, 'reason': 'prototype' if is_prototype else 'no_platform_id'})

    if test_sessions:
        quarantined_workers_detail['test_sessions'] = {
            'reason': 'face experiment with no valid platform ID, or early prototype',
            'sessions': len(test_sessions),
            'files': len(test_sessions),
            'uuids': sorted(e['uuid'] for e in test_sessions),
        }

    # Copy files to the appropriate subfolder
    n_quarantined = 0
    n_userdata = 0

    for filepath in json_files:
        try:
            with open(filepath) as f:
                json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        if filepath in quarantined_paths:
            shutil.copy2(filepath, quarantined_dir)
            n_quarantined += 1
        else:
            shutil.copy2(filepath, userdata_dir)
            n_userdata += 1

    # Write manifest for audit trail
    manifest = {
        'generated': datetime.now().isoformat(),
        'source_directory': directory,
        'max_sessions': max_sessions,
        'total_json_files': len(json_files),
        'quarantined_files': n_quarantined,
        'quarantined_workers': len(quarantined_workers_detail),
        'userdata_files': n_userdata,
        'no_id_files': len(no_id_files),
        'workers': quarantined_workers_detail,
    }
    with open(os.path.join(quarantined_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  Quarantine: {n_quarantined} files from {len(quarantined_workers_detail)} repeat workers → quarantined/")
    print(f"  User data:  {n_userdata} files ({len(no_id_files)} without worker ID) → user-data/")

    return manifest


def _parse_session(raw):
    uuid = raw['UUID']
    data = raw.get('data', {})
    session = data.get('session', {})
    trials = data.get('trials', [])
    return {
        'uuid': uuid,
        'experiment': raw.get('experiment', ''),
        'session_group': session.get('session_group', ''),
        'experiment_version': session.get('experiment_version', ''),
        'n_trials': len(trials),
        'browser': session.get('browserData', {}).get('browser', ''),
    }

def _parse_trials(raw):
    uuid = raw['UUID']
    experiment = raw.get('experiment', '')
    data = raw.get('data', {})
    trials = data.get('trials', [])
    
    session = data.get('session', {})
    worker_id = ''
    if 'mturk' in session and isinstance(session['mturk'], dict):
        worker_id = session['mturk'].get('workerId', '')
    if 'prolific' in session and isinstance(session['prolific'], dict):
        pid = session['prolific'].get('prolificPid', '')
        if pid:
            worker_id = pid
            
    rows = []
    for i, t in enumerate(trials):
        row = {
            'uuid': uuid,
            'experiment': experiment,
            'workerId': worker_id,
            'trial_index': t.get('trial_index', i),
            'rt': t.get('rt', np.nan),
            'correct': t.get('correct', False),
            'response': t.get('response', ''),
            'congruent': t.get('congruent', False),
            'testDirection': t.get('testDirection', np.nan),
        }
        
        if 'bars' in experiment:
            row['adaptorDirection'] = t.get('adaptorDirection', np.nan)
            row['condition'] = 'bars'
        elif 'face' in experiment:
            row['eyesOpen'] = t.get('eyesOpen', True)
            row['faceOnLeft'] = t.get('faceOnLeft', True)
            row['impliedDirection'] = t.get('impliedDirection', np.nan)
            
            gt = t.get('gazeTowards')
            gc = t.get('gazeCondition')
            if gt is False or gc == 'away':
                row['face_direction'] = 'away'
            else:
                row['face_direction'] = 'towards'

            # Unify blindfold/sighted across schema changes:
            #   v1.3-1.4: eyesOpen (True/False)
            #   v1.5-1.7: sighted only (eyesOpen always True)
            #   v1.8.6-1.9.2: sighted-only face-direction runs
            #   v1.10+: condition field ('blindfold'/'normalFace')
            version = session.get('experiment_version', '')
            cond_field = t.get('condition', '')
            if cond_field == 'blindfold':
                row['eyes_condition'] = 'blindfold'
            elif cond_field == 'normalFace':
                row['eyes_condition'] = 'sighted'
            elif 'eyesOpen' in t:
                row['eyes_condition'] = 'sighted' if t['eyesOpen'] else 'blindfold'
            elif version in {'1.8.6', '1.9.2'}:
                row['eyes_condition'] = 'sighted'
            else:
                row['eyes_condition'] = 'unknown'

            row['condition'] = 'faces'
            
        rows.append(row)
    return rows

def load_from_json(directory, experiment_filter=None):
    json_files = sorted(glob.glob(os.path.join(directory, '*.json')))
    if not json_files:
        print(f"  No JSON files found in {directory}")
        return pd.DataFrame(), pd.DataFrame()

    sessions, all_trials = [], []
    seen_uuids = set()

    for filepath in json_files:
        basename = os.path.basename(filepath)
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARNING: Skipping malformed JSON: {basename}")
            continue

        experiment = raw.get('experiment', '')
        if experiment_filter and experiment_filter not in experiment:
            continue

        uuid = raw.get('UUID', '')
        if not uuid or uuid in seen_uuids:
            continue
        
        seen_uuids.add(uuid)
        sessions.append(_parse_session(raw))
        all_trials.extend(_parse_trials(raw))

    sessions_df = pd.DataFrame(sessions)
    trials_df = pd.DataFrame(all_trials)
    
    if trials_df.empty:
        return sessions_df, trials_df
        
    # Unify columns
    uid_map = {u: i + 1 for i, u in enumerate(sorted(trials_df['uuid'].unique()))}
    trials_df['user_number'] = trials_df['uuid'].map(uid_map)
    
    if not sessions_df.empty and 'session_group' in sessions_df.columns:
        sg_map = sessions_df.set_index('uuid')['session_group'].to_dict()
        trials_df['session_group'] = trials_df['uuid'].map(sg_map)
    else:
        trials_df['session_group'] = ''
        
    print(f"  Loaded {len(seen_uuids)} unique participants from {len(json_files)} files")
    return sessions_df, trials_df

def load_star_data():
    """
    Main entry point for loading all star experiment data.

    Runs quarantine_workers() first (copy-based, re-runnable), then loads
    from user-data/ if it exists, otherwise falls back to raw/.
    """
    quarantine_workers(_STAR_DIR)
    
    user_data = os.path.join(_STAR_DIR, 'user-data')
    if os.path.isdir(user_data):
        print("  Loading star data from user-data/ (post-quarantine)")
    else:
        user_data = _STAR_DIR

    sessions_df, trials_df = load_from_json(user_data)
    
    # Validation filters
    if not trials_df.empty:
        # Screen out too fast trials (< 200ms) or obvious botting
        trials_df['valid'] = trials_df['rt'] > 200
        
        # Calculate per-user accuracy
        user_acc = trials_df.groupby('uuid')['correct'].mean().reset_index()
        user_acc.columns = ['uuid', 'overall_accuracy']
        
        trials_df = trials_df.merge(user_acc, on='uuid')
        # Guterstam 2020 commonly excludes participants with < 80% accuracy
        trials_df['high_accuracy'] = trials_df['overall_accuracy'] >= 0.80
        
    return sessions_df, trials_df
