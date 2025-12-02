import pandas as pd
import numpy as np
from scipy import stats
from . import processing

class TubeTrials:
    def __init__(self, data):
        """
        Initialize TubeTrials with data.
        
        Args:
            data (str or pd.DataFrame): Path to CSV file or existing DataFrame.
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Data must be a file path or pandas DataFrame")
            
    def process_angles(self):
        """
        Renames face IDs and calculates end_angle.
        Adds columns: 'end_angle'
        """
        self.df = processing.rename_face_ids(self.df)
        self.df = processing.transform_angles(self.df)
        
    def mark_valid_angles(self, min_angle=3, max_angle=40):
        """
        Marks trials as valid/invalid based on angle range.
        Adds column: 'angle_valid' (bool)
        """
        # processing.validate_angles adds 'angle_valid' column
        self.df = processing.validate_angles(self.df, min_angle, max_angle)
        
    def mark_bad_subjects(self, max_invalid_trials=2):
        """
        Marks subjects to exclude based on invalid trials.
        Adds column: 'is_excluded_subject' (bool)
        """
        if 'angle_valid' not in self.df.columns:
            raise ValueError("Run mark_valid_angles() first.")
            
        bad_subjects = processing.identify_bad_subjects(self.df, max_invalid_trials)
        self.df['is_excluded_subject'] = self.df['user_number'].isin(bad_subjects)
        
    def select(self, valid_only=False, query=None):
        """
        Returns a NEW TubeTrials instance with a subset of data.
        
        Args:
            valid_only (bool): If True, filters by angle_valid=True and is_excluded_subject=False.
            query (str): Pandas query string.
        """
        new_df = self.df.copy()
        
        if valid_only:
            if 'angle_valid' in new_df.columns:
                new_df = new_df[new_df['angle_valid']]
            if 'is_excluded_subject' in new_df.columns:
                new_df = new_df[~new_df['is_excluded_subject']]
                
        if query:
            new_df = new_df.query(query)
            
        return TubeTrials(new_df)
        
    def calc_d_values(self):
        """
        Calculates d values for pairs.
        Returns a pandas DataFrame of pairs.
        """
        # Ensure necessary columns exist
        required_cols = ['end_angle', 'angle_valid']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}'. Run process_angles() and mark_valid_angles() first.")
                
        results_df, _ = processing.balance_trials(self.df)
        return results_df
    
    def get_vector(self, field):
        """
        Returns a numpy array of values for a specific field.
        Useful for plotting.
        """
        if field not in self.df.columns:
            raise ValueError(f"Field '{field}' not found in data.")
        return self.df[field].values
        
    def get_scatter_data(self, x, y):
        """
        Returns a DataFrame with x and y columns for scatter plotting.
        """
        if x not in self.df.columns or y not in self.df.columns:
            raise ValueError(f"Fields '{x}' or '{y}' not found in data.")
        return self.df[[x, y]].copy()
        
    def calc_subject_D(self):
        """
        Calculates 'Big D' (average of 'd' values) per subject per face.
        Returns a DataFrame with columns: user_number, face_id, D
        """
        pairs_df = self.calc_d_values()
        if pairs_df.empty:
            return pd.DataFrame(columns=['user_number', 'face_id', 'D'])
            
        # Group by user and face, calculate mean of 'd'
        subject_D = pairs_df.groupby(['user_number', 'face_id'])['d'].mean().reset_index()
        subject_D.rename(columns={'d': 'D'}, inplace=True)
        return subject_D

    def calc_stats(self):
        """
        Calculates statistics for D values grouped by FaceID.
        Performs a one-sample t-test against 0 for each face,
        using the subject-level average D values.
        Returns a DataFrame with stats.
        """
        subject_D_df = self.calc_subject_D()
        
        if subject_D_df.empty:
            return pd.DataFrame()
            
        stats_results = []
        for face_id in subject_D_df['face_id'].unique():
            face_data = subject_D_df[subject_D_df['face_id'] == face_id]['D']
            if len(face_data) > 1:
                t_stat, p_val = stats.ttest_1samp(face_data, 0)
                stats_results.append({
                    'face_id': face_id,
                    'mean_D': face_data.mean(),
                    'std_D': face_data.std(),
                    'n_subjects': len(face_data),
                    't_stat': t_stat,
                    'p_value': p_val
                })
            else:
                 stats_results.append({
                    'face_id': face_id,
                    'mean_D': face_data.mean() if len(face_data) > 0 else np.nan,
                    'std_D': np.nan,
                    'n_subjects': len(face_data),
                    't_stat': np.nan,
                    'p_value': np.nan
                })
                
        return pd.DataFrame(stats_results)

    def __len__(self):
        return len(self.df)
        
    def __repr__(self):
        return f"<TubeTrials: {len(self.df)} trials>"
