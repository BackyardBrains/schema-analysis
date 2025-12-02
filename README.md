# Schema Analysis Tool

A Python package for analyzing schema experiment data.

## Usage in Google Colab

1.  **Clone the repository** (or upload the `schema_analysis` folder to your Drive).
2.  **Install the package**:
    ```python
    !pip install git+https://github.com/BackyardBrains/schema-analysis.git
    ```
3.  **Run the analysis**:
    ```python
    from schema_analysis import Experiment
    import os

    # Path to your data file (upload it to Colab or mount Drive)
    data_file = "path/to/your/data.csv" 

    # Initialize experiment
    experiment = Experiment(data_file)

    # View results
    print(experiment)
    print(experiment.data.head())
    
    # Access statistics
    print(experiment.stats)
    ```

## Package Structure
- `schema_analysis/experiment.py`: Main entry point (`Experiment` class).
- `schema_analysis/processing.py`: Core data processing logic.
