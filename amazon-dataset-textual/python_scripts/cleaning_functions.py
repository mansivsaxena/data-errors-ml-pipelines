import pandas as pd
import numpy as np
import re
import ftfy
from cleanlab.classification import CleanLearning
import nltk
from nltk.stem import WordNetLemmatizer

# Cleaning functions

def standardize_data_input(df):
    """
    Standardizes data before any cleaning strategy:
    1. Drops rows with missing labels 
    2. Fills missing text with a placeholder
    3. Ensures numeric label types
    """
    df_clean = df.copy()

    df_clean = df_clean.dropna(subset=['label'])

    df_clean['text'] = df_clean['text'].fillna("[missing_text_placeholder]")

    df_clean['label'] = pd.to_numeric(df_clean['label'], errors='coerce')

    df_clean = df_clean.dropna(subset=['label'])
    df_clean['label'] = df_clean['label'].astype(int)

    return df_clean

def clean_basic(df):
    """
    Basic cleaning:
    1. Drop rows with missing labels, fill missing text with placeholder
    2. Standardize text (strip, lowercase, fix encoding)
    3. Keep labels within valid range (1-5)
    4. Convert labels to int
    """
    df = df.copy()

    initial_count = len(df)
    missing_text_count = df['text'].isna().sum()
    initial_labels = df['label'].isna().sum()

    df['text'] = df['text'].replace(r'^\s*$', np.nan, regex=True)
    df = standardize_data_input(df)
    df['text'] = df['text'].apply(
        lambda x: ftfy.fix_text(re.sub(r'\s+', ' ', x.strip().lower()))
    )
    df = df[df['label'].between(1, 5)]
    df['label'] = df['label'].astype(int)

    dropped = initial_count - len(df)

    # return cleaned df and stats
    return df, {
        "strategy_drop": dropped,
        "structural_drop": 0,
        "name": "Basic Cleaning",
        "missing_text": missing_text_count,
        "missing_labels": initial_labels
    }

# Pre-download required assets
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

URL_PATTERN = re.compile(r"http\S+|www\S+")
GARBLED_PATTERN = re.compile(r"[^\x00-\x7F]+")

# Regex needed
NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s']")  # keep apostrophes for negations
MULTI_SPACE = re.compile(r"\s+")

# Common abbreviations map
ABBREV_MAP = {
    r"tbh": "to be honest",
    r"w/": "with",
    r"u": "you",
    r"imo": "in my opinion",
    r"fyi": "for your information",
    r"pls": "please",
    r"thx": "thanks",
    r"plz": "please",
    r"btw": "by the way",
    r"cuz": "because",
    r"cos": "because",
    r"idk": "i do not know",
    r"asap": "as soon as possible",
    r"omg": "oh my god",
    r"lol": "laughing out loud",
    r"ppl": "people",
    r"hrs": "hours",
    r"mins": "minutes",
    r"wks": "weeks"
}

ABBREV_RE = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ABBREV_MAP.keys()) + r')\b', flags=re.IGNORECASE)

def clean_semantic(df):
    """
    Semantic cleaning: Tries to preserve meaning while cleaning
    1. Standardize data input
    2. Lowercase & expand abbreviations
    3. Reduce repeated chars
    4. Remove unwanted chars (except apostrophes)
    5. Normalize whitespace
    6. Token-level negation handling
    7. Lemmatization
    8. Fallback for empty text
    """
    df = df.copy()
    initial_len = len(df)

    df = standardize_data_input(df)
    structural_drop = initial_len - len(df)    

    initial_text_after_std = df['text'].copy()

    # expand abbreviations
    def expand_abbrev(text):
        return ABBREV_RE.sub(lambda m: ABBREV_MAP[m.group(0).lower()], text.lower())
    df['text'] = df['text'].astype(str).map(expand_abbrev)

    # reduce repeated chars ("baaaad" -> "baad")
    df['text'] = df['text'].str.replace(r"(.)\1{2,}", r"\1\1", regex=True)

    # remove unwanted chars (except apostrophes)
    df['text'] = df['text'].map(lambda x: NON_ALPHANUM.sub(" ", x))

    # normalize whitespace
    df['text'] = df['text'].map(lambda x: MULTI_SPACE.sub(" ", x).strip())

    # handle negations instead of processing them separately (e.g., "not good" -> "not_good")
    NEG_WORDS = {
        "not", "no", "never", "none",
        "dont", "didnt", "wasnt", "isnt", "arent", "werent", "cant", "couldnt",
        "don't", "didn't", "wasn't", "isn't", "aren't", "weren't", "can't", "couldn't"
    }
    def join_negations(text):
        tokens = text.split()
        cleaned = []
        i = 0
        while i < len(tokens):
            if tokens[i] in NEG_WORDS and i + 1 < len(tokens):
                cleaned.append(f"{tokens[i]}_{tokens[i+1]}")
                i += 2
            else:
                if len(tokens[i]) > 1:
                    cleaned.append(tokens[i])
                i += 1
        return " ".join(cleaned)
    df['text'] = df['text'].map(join_negations)

    # lemmatization
    df['text'] = df['text'].map(lambda x: " ".join([lemmatizer.lemmatize(w) for w in x.split()]))

    # fallback for empty text
    df['text'] = df['text'].replace("", "neutral_content")

    num_modified = (initial_text_after_std != df['text']).sum()

    return df, {
        "structural_drop": structural_drop, 
        "strategy_drop": 0, 
        "name": "Semantic Cleaning", 
        "modified": num_modified
        }


def clean_heuristic(df):
  """
    Heuristic cleaning: Handle outliers based on heuristics
    1. Drop rows with missing labels
    2. Length filter (drop extremely short or top 1% long spam)
    3. Alpha ratio stricter (catch numeric spam)
    4. Vocabulary richness
    5. Repeated character spam
    6. Drop text that is just URLs or garbled
  """
  df = df.copy()
  initial_count = len(df)

  df = df.dropna(subset=['label'])

  structural_drop = initial_count - len(df)

  text = df["text"].astype(str)

  # drop extremely short or top 1% long spam
  char_len = text.str.len()
  upper_len = char_len.quantile(0.99)
  len_mask = (char_len > 3) & (char_len <= upper_len)

  # alpha ratio (catch numeric spam)
  alpha_ratio = text.str.count(r"[a-zA-Z]") / char_len.replace(0,1)
  alpha_mask = alpha_ratio >= 0.3

  # vocabulary richness
  tokens = text.str.split()
  unique_ratio = tokens.apply(lambda t: len(set(t))/len(t) if len(t)>0 else 0)
  richness_mask = unique_ratio >= 0.05

  # repeated character spam
  repeated_mask = ~text.str.contains(r"(.)\1{10,}", regex=True)

  # text that is just URLs or garbled
  url_mask = ~text.str.match(URL_PATTERN)
  garbled_mask = ~text.str.match(GARBLED_PATTERN)

  final_mask = len_mask & alpha_mask & richness_mask & repeated_mask & url_mask & garbled_mask

  df_clean = df[final_mask]

  dropped = initial_count - len(df_clean)

  return df_clean, {
      "structural_drop": structural_drop, 
      "strategy_drop": dropped, 
      "name": "Heuristic Cleaning"
      }

def clean_model_aware(df, model_pipeline):
    """
    Model-aware cleaning using Cleanlab to identify label issues and assign sample weights accordingly
    1. Standardize data input
    2. Prepare data for Cleanlab
    3. Run Cleanlab's CleanLearning with our model and data to identify label issues
    4. Extract quality scores and issues
    5. Get confidence scores for stats
    6. Assign weights (0.0 to 1.0) based on quality scores
    7. Compile stats
    8. Return full dataframe with new weight column
    """
    initial_count = len(df)
    df = df.copy()

    df = standardize_data_input(df)
    structural_drop = initial_count - len(df)
    valid_count = len(df)

    try:
        if valid_count < 50:
            return df, {"dropped": initial_count - valid_count, "name": "Model-Aware", "stats": None}

        y_data, _ = pd.factorize(df['label'].astype(int), sort=True)
        X_data = df['text'].astype(str).values

        # running CleanLearning with our logistic regression model pipeline
        cl = CleanLearning(clf=model_pipeline, cv_n_folds=3)
        cl.fit(X_data, y_data)

        # quality Scores and issues
        label_issues = cl.get_label_issues()
        is_issue = label_issues['is_label_issue'].values
        quality_scores = label_issues['label_quality'].values
        n_issues = is_issue.sum()

        # confidence scores
        probs = cl.predict_proba(X_data)
        max_conf = probs.max(axis=1)

        # weights (0.0 to 1.0) - higher quality = higher weight
        df['sample_weight'] = quality_scores

        stats = {
            "n_issues": n_issues,
            "issue_rate": n_issues / len(df),
            "avg_conf_clean": max_conf[~is_issue].mean() if any(~is_issue) else 0,
            "avg_conf_noisy": max_conf[is_issue].mean() if any(is_issue) else 0,
            "avg_weight": quality_scores.mean(),
            "class_noise": df.assign(is_issue=is_issue).groupby("label")["is_issue"].mean().to_dict()
        }

        return df, {
            "dropped": initial_count - valid_count, 
            "name": "Model Aware Reweighting",
            "stats": stats
            }
    except Exception as e:
        print(f"Reweighting Error: {e}")
        df['sample_weight'] = 1.0
        return df, {
            "structural_drop": structural_drop, 
            "strategy_drop": 0, 
            "name": "Model-Aware (Failed)", 
            "stats": None
            }
        
print("Cleaning functions loaded.")
