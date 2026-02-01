# Adult Income Dataset - Corruption and Cleaning Evaluation

Evaluates how data corruption affects ML model performance on the UCI Adult Income dataset and tests recovery through traditional and model-aware cleaning strategies.

## Dataset

**Source:** UCI Adult Income Dataset  
**Task:** Binary classification (income <= 50K or > 50K)  
**Features:** Numerical (age, education-num, capital-gain, capital-loss, hours-per-week) and categorical (workclass, marital-status, occupation, relationship, race, sex)  
**Size:** 48,842 samples

## Evaluation Pipeline

The notebook tests two models through four stages:

1. **Baseline** - Clean data performance
2. **Corrupted** - Combined numerical and categorical corruptions
3. **Cleaned (Standard)** - Rule-based cleaning with AutoClean
4. **Cleaned (Model-Aware)** - Standard cleaning + Cleanlab

**Models:**
- LogisticRegression (linear classifier)
- HistGradientBoostingClassifier (tree-based classifier)

**Corruptions Applied:**
- Numerical: Missing values, Gaussian noise, scaling, constraint violations, text injection
- Categorical: Category shifts, typos, missing values

**Cleaning Methods:**
- Salvage numeric columns (extract numbers from corrupted strings)
- Imputation and standard scaling
- AutoClean (outliers, duplicates, categorical imputation)
- Cleanlab (remove mislabeled data with 98% confidence threshold)

## How to Run

### Requirements
- Python 3.11 (required for TensorFlow Data Validation)
- Google Colab (recommended) or local Jupyter environment
- Google Drive (for data persistence in Colab)

### Setup

1. Open notebook in Google Colab
2. Run the setup cells to install dependencies:
```bash
pip install datasets scikit-learn pandas numpy matplotlib
pip install py-AutoClean cleanlab jenga ftfy ucimlrepo category_encoders
```

3. Mount Google Drive (Colab) or set local working directory
4. Run cells sequentially

### TensorFlow Data Validation Setup

TFDV requires specific versions. If using Colab with Python 3.11:
```bash
pip install tensorflow==2.15.1 tensorflow-metadata==1.15.0
pip install tensorflow-data-validation==1.15.1
```

Note: Runtime will restart after TFDV installation. Re-run environment setup cells before continuing.

## Outputs

### Results Files
- `results/ALL_RESULTS_ADULT_COMBINED.csv` - LogisticRegression results
- `results/ALL_RESULTS_ADULT_COMBINED_GB_2.csv` - HistGradientBoosting results
- `results/statistical_tests/` - Statistical analysis outputs (recovery rates, cost-benefit, KS tests)

### Corrupted Data
- `data/corrupted/corrupted_numerical.csv`
- `data/corrupted/corrupted_category_shift.csv`
- `data/corrupted/corrupted_category_typo.csv`
- `data/corrupted/corrupted_missing_values.csv`

### Cleaned Data
- `data/cleaned/cleaned_without_cleanlab.csv`
- `data/cleaned/cleaned_with_cleanlab.csv`

### Figures
- `figures/model_comparison/` - Performance comparison plots
- `figures/cost_benefit/` - Cost-benefit analysis
- `figures/tfdv_analysis/` - TFDV detection comparison

**Final-Takeaways:** Model-aware cleaning helps linear models but harms tree-based models. Tree-based models handle corrupted data naturally and removing data degrades their performance.

## Repository Structure

```
adults-dataset-categorical-numerical/
├── adult_data_clean_corrupt_eval.ipynb
├── data/
│   ├── baseline/
│   ├── corrupted/
│   └── cleaned/
├── results/
│   └── statistical_tests/
├── figures/
│   ├── model_comparison/
│   ├── cost_benefit/
│   └── tfdv_analysis/
└── python_scripts/
    ├── inject.py
    └── clean_num.py
```

## Notes

- HistGradientBoostingClassifier handles NaNs natively, no imputation needed
- Fixed random seeds (42) for reproducibility
- Multiple runs per configuration for robust statistical validation
- TFDV analysis requires Python 3.11 specifically
