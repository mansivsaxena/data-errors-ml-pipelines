from jenga.corruptions.generic import MissingValues
from jenga.corruptions.text import BrokenCharacters
import numpy as np

# Corruption functions

def apply_missing_text(df, fraction=0.30):
    """01: Missing Values in text - 30%"""
    df = df.copy()
    mv = MissingValues(column="text", fraction=fraction, missingness="MCAR")
    return mv.transform(df)

def apply_broken_characters(df, fraction=0.25):
    """02: Broken Characters in text - 25%"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=fraction)
    return bc.transform(df)

def apply_swapped_text(df, fraction=0.20):
    """03: Swapped text values - 20% (Custom function)"""
    df = df.copy()
    np.random.seed(42)

    n_rows = len(df)
    n_swap = int(fraction * n_rows)

    # Select rows to swap
    swap_idx = np.random.choice(df.index, size=n_swap, replace=False)
    shuffled_idx = np.random.permutation(swap_idx)

    # Swap 'text' values across rows (stays within text column)
    df.loc[swap_idx, 'text'] = df.loc[shuffled_idx, 'text'].values

    return df

def apply_missing_labels(df, fraction=0.15):
    """04: Missing Labels - 15%"""
    df = df.copy()
    mv = MissingValues(column="label", fraction=fraction, missingness="MCAR")
    return mv.transform(df)

def apply_swapped_labels(df, fraction=0.12):
    """05: Swapped Labels - 12% (Custom function)"""
    df = df.copy()
    np.random.seed(42)

    n_rows = len(df)
    n_swap = int(fraction * n_rows)

    # Select rows to swap
    swap_idx = np.random.choice(df.index, size=n_swap, replace=False)
    shuffled_idx = np.random.permutation(swap_idx)

    # Swap 'label' values across rows (stays within label column)
    df.loc[swap_idx, 'label'] = df.loc[shuffled_idx, 'label'].values

    return df

def apply_combined_broken_chars_missing_text(df):
    """06: Broken Chars (10%) + Missing (8%)"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=0.10)
    df = bc.transform(df)
    mv = MissingValues(column="text", fraction=0.08, missingness="MCAR")
    return mv.transform(df)

def apply_combined_swap_text_labels(df):
    """07: Swapped Text (15%) + Swapped Labels (8%)"""
    df = df.copy()

    # Swap 15% of text
    df = apply_swapped_text(df, fraction=0.15)
    # Swap 8% of labels
    df = apply_swapped_labels(df, fraction=0.08)

    return df

def apply_heavy_missing(df):
    """08: Heavy Missing - Text (25%) + Labels (10%)"""
    df = df.copy()
    mv_text = MissingValues(column="text", fraction=0.25, missingness="MCAR")
    df = mv_text.transform(df)
    mv_label = MissingValues(column="label", fraction=0.10, missingness="MCAR")
    return mv_label.transform(df)

def apply_all_corruptions(df):
    """09: All - Broken (8%) + Swapped (10%) + Missing (5%)"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=0.08)
    df = bc.transform(df)
    df = apply_swapped_text(df, fraction=0.10)
    mv_text = MissingValues(column="text", fraction=0.05, missingness="MCAR")
    df = mv_text.transform(df)
    df = apply_swapped_labels(df, fraction=0.05)
    return df

print("Corruption functions loaded.")