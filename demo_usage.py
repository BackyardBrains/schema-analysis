import sys
import os

# Mimic installing the package by adding it to path
sys.path.append(os.path.abspath('.'))

from schema_analysis import Experiment

# Path to data
DATA_FILE = os.path.join('data', 'facetip_data_Nov2025.csv')

print("--- Starting Demo Analysis ---")
if not os.path.exists(DATA_FILE):
    print(f"WARNING: Data file not found at {DATA_FILE}. Please check the path.")
else:
    # 1. Initialize Experiment (Loads data only)
    experiment = Experiment(DATA_FILE)
    print(f"Initial state: {experiment}")
    
    # 2. Preprocess (Rename faces, calculate angles)
    experiment.preprocess()
    
    # 3. Filter Trials (Define valid angle range)
    experiment.filter_trials(min_angle=3, max_angle=40)
    
    # 4. Exclude Subjects (Define max invalid trials)
    experiment.exclude_subjects(max_invalid_trials=2)
    
    # 5. Balance Trials (Generate final results)
    results = experiment.balance_trials()
    
    if results is not None:
        print("\nFirst 5 rows of results:")
        print(results.head())
        
        print("\nFinal Subjects:")
        print(experiment.subjects)
