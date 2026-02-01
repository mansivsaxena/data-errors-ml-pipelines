# Amazon Product Reviews Dataset - Corruption and Cleaning Evaluation

Evaluates how data corruption affects ML model performance on Amazon product reviews text data and tests recovery through progressive cleaning strategies.

## Dataset

**Source:** Amazon Beauty Reviews 2023  
**Task:** Multi-class classification (rating 1-5 prediction from review text)  
**Features:** Review text (unstructured text data)  
**Size:** 100,000 samples

## Evaluation Pipeline

The notebook tests a single model through four cleaning techniques:

1. **Baseline** - Clean data performance
2. **Basic Cleaning** - Drop nulls, standardize text, fix encoding
3. **Heuristic Cleaning** - Remove spam, URLs, garbled text, extreme lengths
4. **Semantic Cleaning** - Expand abbreviations, lemmatization, negation handling
5. **Model-Aware Cleaning** - Semantic cleaning + Cleanlab reweighting

**Model:**
- LogisticRegression with dual TF-IDF (word and character n-grams)

**Corruptions Applied:**
- Missing text (30% of reviews)
- Broken characters (25% - character replacements like a → á)
- Swapped text (20% - row value swaps)
- Missing labels (15% of ratings)
- Swapped labels (12% - rating swaps)
- Combined corruptions (broken chars + missing text, swapped text + labels)
- Heavy missing (25% text + 10% labels)
- All corruptions combined

**Cleaning Techniques:**
- **Basic:** Drop missing ratings, fill missing text, standardize encoding, lowercase, valid range enforcement
- **Heuristic:** Remove URLs, spam detection, length filtering, garbled text removal
- **Semantic:** Abbreviation expansion, repeated character reduction, negation handling, lemmatization
- **Model-Aware:** L3 + Cleanlab confidence-based sample reweighting

## Getting Started

### Clone Repository

```bash
git clone https://github.com/mansivsaxena/data-errors-ml-pipelines.git
cd data-errors-ml-pipelines/amazon-dataset-textual
```

### Requirements
- Python 3.11 (required for TensorFlow Data Validation)
- Google Colab (recommended) or local Jupyter environment
- Google Drive (for data persistence in Colab)

### Setup

1. Open `amazon_data_clean_corrupt_eval.ipynb` in Google Colab
2. Run the setup cells to install dependencies:
```bash
pip install datasets scikit-learn pandas numpy matplotlib
pip install jenga ftfy cleanlab
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

For TFDV analysis, open `amazon_data_tfdv_analysis.ipynb` after completing the main evaluation notebook.

## Outputs

### Results Files
- `results/all_results_combined.csv` - Combined results from all experiments
- `results/baseline_results.csv` - Baseline performance
- `results/batch1_results.csv`, `results/batch2_results.csv` - Corruption batch results
- `results/all_cleanlab_stats_combined.csv` - Cleanlab analysis results
- `results/stats/statistical_tests/` - Statistical test outputs
- `results/stats/tables/` - Analysis tables

### Corrupted Data
- `data/01_missing_text_data.csv`
- `data/02_broken_chars_data.csv`
- `data/03_swapped_text_data.csv`
- `data/04_missing_labels_data.csv`
- `data/05_swapped_labels_data.csv`
- `data/06_combined_broken_chars_missing_text_data.csv`
- `data/07_combined_swap_text_labels_data.csv`
- `data/08_heavy_missing_data.csv`
- `data/09_all_corruptions_data.csv`
- `data/baseline_data.csv`

### TFDV Analysis
- `results/tfdv_eval/anomalies_*.pbtxt` - Anomaly reports per corruption
- `results/tfdv_eval/baseline_schema.pbtxt` - Baseline schema
- `results/tfdv_eval/tfdv_analysis.csv` - TFDV detectability scores

### Figures
- `figures/performance_analysis/` - Corruption damage, recovery trends, accuracy distributions
- `figures/computation_analysis/` - Time breakdowns, overhead ratios, cleaning costs
- `figures/cleanlab_analysis/` - Confidence separation, noise heatmaps, reweighting behavior

**Key-Takeaways:** Model-aware cleaning outperforms traditional methods for text data. Semantic corruptions escape schema validation but significantly impact model performance, justifying model-aware approaches.

## Repository Structure

```
amazon-dataset-textual/
├── amazon_data_clean_corrupt_eval.ipynb
├── amazon_data_tfdv_analysis.ipynb
├── data/
│   ├── baseline_data.csv
│   └── [9 corrupted datasets]
├── results/
│   ├── stats/
│   │   ├── statistical_tests/
│   │   └── tables/
│   ├── tfdv_eval/
│   └── [result CSVs]
├── figures/
│   ├── performance_analysis/
│   ├── computation_analysis/
│   └── cleanlab_analysis/
└── python_scripts/
    ├── cleaning_functions.py
    └── corruptions.py
```

## Notes

- TF-IDF uses both word n-grams (1-2) and character n-grams (3-5) for robust text representation
- Fixed random seeds (42) for reproducibility
- Cleanlab uses confidence scores to reweight training samples, not remove them
- TFDV analysis requires Python 3.11 specifically
- Analysis conducted in batches (batch1, batch2) then combined for final results
