import sys
import os

# Mimic installing the package by adding it to path
sys.path.append(os.path.abspath('.'))

from schema_analysis import TubeTrials

# Path to data
DATA_FILE = os.path.join('data', 'facetip_data_Nov2025.csv')

print("--- Starting Demo Analysis ---")
if not os.path.exists(DATA_FILE):
    print(f"WARNING: Data file not found at {DATA_FILE}. Please check the path.")
else:
    # 1. Initialize TubeTrials
    trials = TubeTrials(DATA_FILE)
    print(f"Initial state: {trials}")
    
    # 2. Process & Tag (Rename faces, calculate angles, mark valid/invalid)
    print("Processing and tagging data...")
    trials.process_angles()
    trials.mark_valid_angles(min_angle=3, max_angle=40)
    trials.mark_bad_subjects(max_invalid_trials=2)
    
    # 3. Select (Filter to get clean data)
    print("Selecting valid trials...")
    clean_trials = trials.select(valid_only=True)
    print(f"Clean state: {clean_trials}")
    
    # 4. Calculate D Values
    print("Calculating D values...")
    results = clean_trials.calc_d_values()
    
    if results is not None:
        print("\nFirst 5 rows of results:")
        print(results.head())
        
        # Example of getting data for plotting
        print("\nExample data for plotting (end_angle):")
        angles = clean_trials.get_vector('end_angle')
        print(f"Angle stats: min={angles.min()}, max={angles.max()}, mean={angles.mean():.2f}")

    # 5. Calculate Stats
    print("\nCalculating Statistics (t-test against 0 per FaceID)...")
    stats_df = clean_trials.calc_stats()
    print(stats_df)
