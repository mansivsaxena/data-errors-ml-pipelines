## UvA Data Preparation 2026 - Team 17

## Data Errors in ML Pipelines

Different data types suffer from corruption specific to each type. For example, negation errors are specific to numeric data. These data corruptions can affect an ML model’s performance to varying degrees. Since different data types can suffer from different corruptions, the effectiveness of cleaning techniques, whether it be model accuracy improvement, time taken to clean, or rows preserved, can vary depending on the type of corruption. Some cleaning techniques are also data type specific, so they cannot be compared directly. Thus, it would be difficult to know which cleaning techniques are worth using depending on your data. This project aims to bridge that knowledge gap, helping developers save time and resources when cleaning their datasets before training their models.

## Overview

This project follows the ‘ML Pipelines and Erroneous Data’ theme as a part of the Data Preparation course at UvA. It investigates the impact of data corruptions and cleaning techniques on machine learning pipelines. The analysis evaluates various corruption types, cleaning methods, and their effects on model performance, computational costs, and data quality.

## Key Features

- **Data Corruption Simulation**: Implements various types of data corruptions to simulate real-world data quality issues
- **Cleaning Method Evaluation**: Compares different data cleaning approaches including Cleanlab-based methods
- **Performance Analysis**: Evaluates impact on ML model accuracy, precision, recall, and F1-score
- **Computational Cost Analysis**: Measures the computational efficiency of different cleaning strategies

## Dataset

Two datasets were in this project, each being used to test different types of data corruption and cleaning techniques
- **Amazon Reviews**: The dataset consists of product reviews from Amazon in 2023. For this dataset, we investigated textual corruptions and the appropriate cleaning techniques. We used the review text to predict the rating (1-5) for the product.

- **Adult Income**: The Adult Income dataset includes a binary classification of whether an individual’s annual salary is over 50,000, along with their personal information. This dataset was used to investigate categorical and numerical corruptions along with the appropriate cleaning techniques. This project uses the  numerical features: ‘age’, ‘educational-num’, ‘capital-gain’, ‘capital-loss’, ‘hours-per-week’, and categorical features: ‘workclass’, ‘marital-status’, ‘occupation’, ‘relationship’, ‘race’, ‘gender’ in order to predict if the individual’s annual salary is over 50,000.

## Methodology

1. **Data Preparation**: Load and preprocess the datasets
2. **Train Baseline**: Train the model on the raw data
3. **Corruption Injection**: Apply various corruption types to simulate data quality issues
4. **Train Corrupted**: Train the model on the corrupted data
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
   # For the Amazon Reviews dataset
   jupyter notebook amazon-dataset-textual/notebook/amazon_data_clean_corrupt_eval.ipynb
   ```

   ```bash
   # For the Adult Income dataset
   jupyter notebook adults-dataset-categorical-numerical/notebook/adult_data_clean_corrupt_eval.ipynb
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