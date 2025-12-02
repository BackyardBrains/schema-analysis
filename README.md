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
    from schema_analysis import Experiment
    import os

    # Path to your data file (upload it to Colab or mount Drive)
    data_file = "path/to/your/data.csv" 

    # Initialize experiment (loads data only)
    experiment = Experiment(data_file)

    # Run processing steps explicitly
    experiment.preprocess()
    experiment.filter_trials(min_angle=3, max_angle=40)
    experiment.exclude_subjects(max_invalid_trials=2)
    
    # Generate results
    results = experiment.balance_trials()

    # View results
    print(results.head())
    ```

## Package Structure
- `schema_analysis/experiment.py`: Main entry point (`Experiment` class).
- `schema_analysis/processing.py`: Core data processing logic.
