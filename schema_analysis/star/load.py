"""
Data loading for star experiments (Implied Motion / RDK).

Parses both `rdk-bars` (Exp1 / adaptation control) and `rdk-face-1` (Exp2 / social attention).
"""

import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STAR_DIR = os.path.join(ROOT, 'data', 'star', 'raw')

def quarantine_workers(directory):
    """
    Classify JSON files into 'quarantined/' (repeat workerId/prolificPid) and 'user-data/'.
    Does not run if user-data already exists to prevent repeated shifting.
    """
    if not os.path.exists(directory):
        return
        
    user_data_dir = os.path.join(directory, 'user-data')
    quarantine_dir = os.path.join(directory, 'quarantined')
    
    if os.path.exists(user_data_dir):
        print(f"  {user_data_dir} exists, skipping quarantine step.")
        return

    os.makedirs(user_data_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(directory, '*.json'))
    
    # 1. First pass: count sessions per worker/prolific ID
    worker_counts = defaultdict(int)
    prolific_counts = defaultdict(int)
    
    for filepath in json_files:
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            continue
            
        session = raw.get('data', {}).get('session', {})
        
        # Check standard mturk workerId
        mturk = session.get('mturk', {})
        if isinstance(mturk, dict):
            wid = mturk.get('workerId', '').strip()
            if wid and 'test' not in wid.lower():
                worker_counts[wid] += 1
                
        # Check prolific PID
        prolific = session.get('prolific', {})
        if isinstance(prolific, dict):
            pid = prolific.get('prolificPid', '').strip()
            if pid and 'test' not in pid.lower():
                prolific_counts[pid] += 1

    # 2. Second pass: Move files based on repeat status
    moved_quarantine = 0
    moved_user = 0
    
    for filepath in json_files:
        basename = os.path.basename(filepath)
        
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            continue
            
        session = raw.get('data', {}).get('session', {})
        
        wid = ''
        mturk = session.get('mturk', {})
        if isinstance(mturk, dict):
            wid = mturk.get('workerId', '').strip()
            
        pid = ''
        prolific = session.get('prolific', {})
        if isinstance(prolific, dict):
            pid = prolific.get('prolificPid', '').strip()

        is_repeat = False
        if wid and 'test' not in wid.lower() and worker_counts[wid] > 1:
            is_repeat = True
        if pid and 'test' not in pid.lower() and prolific_counts[pid] > 1:
            is_repeat = True
            
        dest_dir = quarantine_dir if is_repeat else user_data_dir
        shutil.move(filepath, os.path.join(dest_dir, basename))
        
        if is_repeat:
            moved_quarantine += 1
        else:
            moved_user += 1
            
    print(f"  Quarantine: {moved_user} user-data, {moved_quarantine} quarantined.")


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
            if gt is True or gc == 'towards':
                row['gazeTowards'] = True
            elif gt is False or gc == 'away':
                row['gazeTowards'] = False
            else:
                row['gazeTowards'] = True # fallback for V1 versions without backward faces
                
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
    Automatically handles quarantine logic.
    """
    quarantine_workers(_STAR_DIR)
    
    user_data = os.path.join(_STAR_DIR, 'user-data')
    if not os.path.exists(user_data):
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
