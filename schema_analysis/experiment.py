import pandas as pd
import os
from . import processing

class Experiment:
    def __init__(self, data_path):
        """
        Initialize the Experiment with a data file.
        Only loads the raw data. Call processing methods to filter and analyze.
        
        Args:
            data_path (str): Path to the CSV data file.
        """
        self.data_path = data_path
        self.raw_data = None
        self.data = None # Working copy of data
        self.stats = {}
        
        self._load_data()
        
    def _load_data(self):
        print(f"Loading data from {self.data_path}...")
        try:
            self.raw_data = pd.read_csv(self.data_path)
            self.data = self.raw_data.copy()
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found.")
            return

        print(f"Loaded {len(self.raw_data)} trials from {self.raw_data['user_number'].nunique()} subjects")
        
    def preprocess(self):
        """
        Performs basic data cleanup:
        1. Renames face IDs
        2. Calculates end_angle
        """
        if self.data is None:
            print("No data loaded.")
            return

        print("\n[Preprocessing] Renaming faces and transforming angles...")
        self.data = processing.rename_face_ids(self.data)
        self.data = processing.transform_angles(self.data)
        print("Preprocessing complete.")

    def filter_trials(self, min_angle=3, max_angle=40):
        """
        Marks trials as valid/invalid based on angle range.
        """
        if self.data is None:
            print("No data loaded.")
            return

        print(f"\n[Filtering] Validating trials with angle range ({min_angle}, {max_angle})...")
        self.data = processing.validate_angles(self.data, min_angle, max_angle)
        
        trials_invalidated_angle = len(self.data) - self.data['angle_valid'].sum()
        self.stats['trials_invalidated_angle'] = trials_invalidated_angle
        print(f"Trials invalidated by angle rule: {trials_invalidated_angle}")

    def exclude_subjects(self, max_invalid_trials=2):
        """
        Excludes subjects with too many invalid trials.
        """
        if self.data is None:
            print("No data loaded.")
            return
            
        if 'angle_valid' not in self.data.columns:
            print("Error: Run filter_trials() first.")
            return

        print(f"\n[Exclusion] Excluding subjects with >{max_invalid_trials} invalid trials...")
        self.data, excluded_subjects = processing.exclude_subjects(self.data, max_invalid_trials)
        self.stats['excluded_subjects'] = excluded_subjects
        print(f"Subjects excluded: {len(excluded_subjects)}")

    def balance_trials(self):
        """
        Runs the pairing logic to generate final results.
        Returns the results DataFrame.
        """
        if self.data is None:
            print("No data loaded.")
            return None

        print("\n[Balancing] Identifying valid pairs...")
        results_df, used_indices = processing.balance_trials(self.data)
        self.stats['trials_used_in_pairs'] = len(used_indices)
        print(f"Valid pairs found: {len(results_df)}")
        return results_df
        
    def __repr__(self):
        if self.data is None:
            return f"<Experiment: No data loaded>"
        return f"<Experiment: {len(self.data)} trials (current), {self.data['user_number'].nunique()} subjects>"

    @property
    def subjects(self):
        if self.data is not None:
            return sorted(self.data['user_number'].unique())
        return []
