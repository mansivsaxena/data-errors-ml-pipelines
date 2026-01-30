## UvA Data Preparation 2026 - Team 17

## Data Errors in ML Pipelines

## Overview

This project investigates the impact of data corruptions and cleaning techniques on machine learning pipelines. The analysis evaluates various corruption types, cleaning methods, and their effects on model performance, computational costs, and data quality.

## Key Features

- **Data Corruption Simulation**: Implements various types of data corruptions to simulate real-world data quality issues
- **Cleaning Method Evaluation**: Compares different data cleaning approaches including Cleanlab-based methods
- **Performance Analysis**: Evaluates impact on ML model accuracy, precision, recall, and F1-score
- **Computational Cost Analysis**: Measures the computational efficiency of different cleaning strategies

## Dataset


## Methodology

1. **Data Preparation**: Load and preprocess the datasets
2. **Train Baseline**: Train the model on the raw data
3. **Corruption Injection**: Apply various corruption types to simulate data quality issues
4. **Train Corrupted**: Train the model on the corrupted datat
3. **Cleaning Application**: Use different cleaning methods including baseline and Cleanlab approaches
4. **Model Training**: Train ML models on cleaned datasets
5. **Evaluation**: Compare performance metrics, computational costs, and data retention rates

## Results

The `results/` directory contains comprehensive analysis results including:

- **Cleanlab Statistics**: Detailed statistics from Cleanlab cleaning processes
- **Performance Metrics**: Accuracy, precision, recall, and F1-scores for different methods
- **Computational Analysis**: Time and resource usage comparisons
- **Statistical Tables**: Method-wise analysis, corruption severity impact, and interaction effects

Key findings are visualized in the `figures/` directory with plots for:
- Cleanlab analysis results
- Computational cost breakdowns
- Performance comparisons across methods

## Usage

### Prerequisites

### Running the Analysis

1. **Clone the repository** (if applicable) or navigate to the project directory
2. **Install dependencies**
3. **Run the main evaluation notebook**:
   ```bash
   jupyter notebook amazon-dataset-textual/notebook/amazon_data_clean_corrupt_eval.ipynb
   ```
4. **Execute the notebook cells** to reproduce the analysis

### Using Python Scripts

- `cleaning_functions.py`: Contains utility functions for data cleaning
- `corruptions.py`: Functions to simulate data corruptions

You can import and use these in your own scripts:

```python
from python_scripts.cleaning_functions import clean_data
from python_scripts.corruptions import add_noise
```

## Dependencies

- pandas
- numpy
- scikit-learn
- cleanlab
- matplotlib
- seaborn
- jupyter