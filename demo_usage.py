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
    # Initialize Experiment
    experiment = Experiment(DATA_FILE)
    
    # Show results
    print("\nExperiment Summary:")
    print(experiment)
    
    if experiment.data is not None:
        print("\nFirst 5 rows of processed data:")
        print(experiment.data.head())
        
        print("\nSubjects:")
        print(experiment.subjects)
