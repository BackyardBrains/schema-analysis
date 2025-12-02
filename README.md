# Schema Analysis Tool

A Python package for analyzing schema experiment data.

## Usage in Google Colab

1.  **Clone and install**:
    ```python
    !git clone https://github.com/BackyardBrains/schema-analysis.git
    !pip install ./schema-analysis
    ```
3.  **Run the analysis**:
    ```python
    from schema_analysis import TubeTrials
    import os

    # Path to your data file (upload it to Colab or mount Drive)
    data_file = "path/to/your/data.csv" 

    # 1. Initialize (loads data)
    trials = TubeTrials(data_file)

    # 2. Process & Tag (adds columns, doesn't drop rows)
    trials.process_angles()
    trials.mark_valid_angles(min_angle=3, max_angle=40)
    trials.mark_bad_subjects(max_invalid_trials=2)

    # 3. Select (returns a new TubeTrials instance with filtered data)
    clean_trials = trials.select(valid_only=True)
    
    # 4. Analyze
    # Get arrays for plotting
    angles = clean_trials.get_vector('end_angle')
    
    # Calculate d values for pairs (returns DataFrame with 'd')
    pairs = clean_trials.calc_d_values()

    # Calculate statistics (returns DataFrame)
    # Aggregates 'd' to subject-level 'D' and runs t-test against 0
    stats = clean_trials.calc_stats()

    # View results
    print(stats)
    ```

## Package Structure
- `schema_analysis/tube_trials.py`: Main entry point (`TubeTrials` class).
- `schema_analysis/processing.py`: Core data processing logic.
