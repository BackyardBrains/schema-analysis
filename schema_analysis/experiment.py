import pandas as pd
import os
from . import processing

class Experiment:
    def __init__(self, data_path):
        """
        Initialize the Experiment with a data file.
        Automatically loads and processes the data.
        
        Args:
            data_path (str): Path to the CSV data file.
        """
        self.data_path = data_path
        self.raw_data = None
        self.data = None # Final processed data
        self.stats = {}
        
        self._load_and_process()
        
    def _load_and_process(self):
        print(f"Loading data from {self.data_path}...")
        try:
            self.raw_data = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found.")
            return

        print(f"Loaded {len(self.raw_data)} trials from {self.raw_data['user_number'].nunique()} subjects")
        
        # Pipeline
        df = self.raw_data.copy()
        
        # 1. Rename Face IDs
        df = processing.rename_face_ids(df)
        
        # 2. Transform Angles
        df = processing.transform_angles(df)
        
        # 3. Validate Angles
        df = processing.validate_angles(df)
        trials_invalidated_angle = len(df) - df['angle_valid'].sum()
        self.stats['trials_invalidated_angle'] = trials_invalidated_angle
        print(f"Trials invalidated by angle rule: {trials_invalidated_angle}")
        
        # 4. Exclude Subjects
        df, excluded_subjects = processing.exclude_subjects(df)
        self.stats['excluded_subjects'] = excluded_subjects
        print(f"Subjects excluded: {len(excluded_subjects)}")
        
        # 5. Balance Trials
        self.data, used_indices = processing.balance_trials(df)
        self.stats['trials_used_in_pairs'] = len(used_indices)
        print(f"Valid pairs found: {len(self.data)}")
        
    def __repr__(self):
        if self.data is None:
            return f"<Experiment: No data loaded>"
        return f"<Experiment: {len(self.data)} pairs, {self.data['user_number'].nunique()} subjects>"

    @property
    def subjects(self):
        if self.data is not None:
            return sorted(self.data['user_number'].unique())
        return []
