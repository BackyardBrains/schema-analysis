import pandas as pd
import numpy as np

def rename_face_ids(df):
    """
    Renames specific face IDs based on predefined rules.
    ID001 -> ID017
    ID022 -> ID015
    """
    print(f"Original face IDs: {sorted(df['face_id'].unique())}")
    df['face_id'] = df['face_id'].replace({
        'ID001': 'ID017',
        'ID022': 'ID015'
    })
    print(f"Updated face IDs: {sorted(df['face_id'].unique())}")
    return df

def transform_angles(df):
    """
    Creates 'end_angle' column based on 'tip_direction' and 'raw_angle'.
    Left: raw_angle * -1
    Right: raw_angle * 1
    """
    def calculate_end_angle(row):
        if row['tip_direction'] == 'left':
            return row['raw_angle'] * -1
        elif row['tip_direction'] == 'right':
            return row['raw_angle'] * 1
        return row['raw_angle']
    
    df['end_angle'] = df.apply(calculate_end_angle, axis=1)
    return df

def validate_angles(df, min_angle=3, max_angle=40):
    """
    Marks trials as valid/invalid based on angle range.
    """
    df['angle_valid'] = (df['end_angle'] > min_angle) & (df['end_angle'] < max_angle)
    return df

def identify_bad_subjects(df, max_invalid_trials=2):
    """
    Identifies subjects who have more than `max_invalid_trials` invalid trials.
    Returns a list of subject IDs to exclude.
    """
    angle_invalid_per_subject = df[~df['angle_valid']].groupby('user_number').size()
    subjects_to_exclude = angle_invalid_per_subject[angle_invalid_per_subject > max_invalid_trials].index.tolist()
    return subjects_to_exclude

def exclude_subjects(df, max_invalid_trials=2):
    """
    Excludes subjects who have more than `max_invalid_trials` invalid trials.
    Returns the filtered DataFrame and the list of excluded subjects.
    """
    subjects_to_exclude = identify_bad_subjects(df, max_invalid_trials)
    df_filtered = df[~df['user_number'].isin(subjects_to_exclude)].copy()
    return df_filtered, subjects_to_exclude

def balance_trials(df):
    """
    Identifies valid pairs of trials (FaceLeft and FaceRight) for each user and face.
    Returns a DataFrame of valid D values.
    """
    valid_d_values = []
    used_indices = set()
    
    for user_id in df['user_number'].unique():
        user_df = df[df['user_number'] == user_id]
        for face_id in user_df['face_id'].unique():
            face_df = user_df[user_df['face_id'] == face_id]
            for tube_idx in face_df['tubeTypeIndex'].unique():
                tube_df = face_df[face_df['tubeTypeIndex'] == tube_idx]
                
                # Pair 1: Face Left (LL towards, LR away)
                towards_left = tube_df[(tube_df['faceSide'] == 'left') & (tube_df['tip_direction'] == 'left')]
                away_left = tube_df[(tube_df['faceSide'] == 'left') & (tube_df['tip_direction'] == 'right')]
                
                if len(towards_left) == 1 and len(away_left) == 1:
                    t_row, a_row = towards_left.iloc[0], away_left.iloc[0]
                    if t_row['angle_valid'] and a_row['angle_valid']:
                        d_val = abs(t_row['end_angle']) - abs(a_row['end_angle'])
                        valid_d_values.append({
                            'user_number': user_id,
                            'face_id': face_id,
                            'tubeTypeIndex': tube_idx,
                            'pair_type': 'FaceLeft',
                            'd': d_val
                        })
                        used_indices.update([t_row.name, a_row.name])
                        
                # Pair 2: Face Right (RR towards, RL away)
                towards_right = tube_df[(tube_df['faceSide'] == 'right') & (tube_df['tip_direction'] == 'right')]
                away_right = tube_df[(tube_df['faceSide'] == 'right') & (tube_df['tip_direction'] == 'left')]
                
                if len(towards_right) == 1 and len(away_right) == 1:
                    t_row, a_row = towards_right.iloc[0], away_right.iloc[0]
                    if t_row['angle_valid'] and a_row['angle_valid']:
                        d_val = abs(t_row['end_angle']) - abs(a_row['end_angle'])
                        valid_d_values.append({
                            'user_number': user_id,
                            'face_id': face_id,
                            'tubeTypeIndex': tube_idx,
                            'pair_type': 'FaceRight',
                            'd': d_val
                        })
                        used_indices.update([t_row.name, a_row.name])
                        
    if not valid_d_values:
        results_df = pd.DataFrame(columns=['user_number', 'face_id', 'tubeTypeIndex', 'pair_type', 'd'])
    else:
        results_df = pd.DataFrame(valid_d_values)
    return results_df, used_indices
