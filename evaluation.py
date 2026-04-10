"""

            Calculate the metrics for the evaluation of the model. Of interest are the following metrics:

            - Content Integrity
            - Content Ratio
            - Content Type
            - Business Sector
            - Technical Content
            - Content Quality
            - Information Density
            - Educational Value
            - Reasoning Indicators
            - Audience Level
            - Commercial Bias
            - Time-Sensitivity
            - Content Safety
            - PII Presence
            - Regional Relevance
            - Country Relevance

        Ordinal Properties: Quadratic Weighted Kappa (QWK)
        Binary Property: F1 Score
        Multi-select Property: Jaccard Similarity Coefficient (JSC)

"""

import sklearn.metrics as metrics
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
import os


GOLD_PATH = "test_01.jsonl"
PREDICTIONS_PATH_1 = "test_01_propella_ordered.jsonl"
PREDICTIONS_PATH_2 = "test_01_adapter_ordered.jsonl"
PREDICTIONS_PATH_3 = "test_01_latxa_ordered.jsonl"


def calculate_qwk(y_true, y_pred):
    """
    Calculate the Quadratic Weighted Kappa (QWK) for ordinal properties.
    """
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    if len(unique_true) == len(unique_pred) or not np.array_equal(unique_true, unique_pred):
        if np.all(y_true == y_pred):
            return 1.0
    return metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 Score for binary properties.
    """
    return metrics.f1_score(y_true, y_pred)

def calculate_jaccard_similarity(y_true, y_pred):
    """
    Calculate the Jaccard Similarity Coefficient (JSC) for multi-select properties.
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred)
    y_true_binarized = mlb.transform(y_true)
    y_pred_binarized = mlb.transform(y_pred)
    return metrics.jaccard_score(y_true_binarized, y_pred_binarized, average='samples')

def get_encoder(feature_name):
    return OrdinalEncoder(categories=[configs[feature_name]])

def load_jsonl(path):
    """Parses a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

if __name__ == "__main__":

    configs = {
        'content_integrity': ['severely_degraded', 'fragment', 'mostly_complete', 'complete'],
        'content_ratio': ['minimal_content', 'mostly_navigation', 'mixed_content', 'mostly_content',
                          'complete_content'],
        'content_length': ['minimal', 'brief', 'moderate', 'substantial'],
        'content_quality': ['unacceptable', 'poor', 'adequate', 'good', 'excellent'],
        'information_density': ['empty', 'thin', 'moderate', 'adequate', 'dense'],
        'educational_value': ['none', 'minimal', 'basic', 'moderate', 'high'],
        'reasoning_indicators': ['none', 'minimal', 'basic_reasoning', 'explanatory', 'analytical'],
        'audience_level': ['children', 'youth', 'beginner', 'general', 'advanced', 'expert'],
        'commercial_bias': ['none', 'minimal', 'moderate', 'heavy', 'pure_marketing'],
        'time_sensitivity': ['time_sensitive', 'regularly_updating', 'slowly_changing', 'evergreen'],
        'content_safety': ['safe', 'mild_concerns', 'nsfw', 'harmful', 'illegal']
    }



    # Example transformation

    # Load the gold annotations and predictions
    with open(GOLD_PATH, 'r') as f:
        gold_data = [json.loads(line) for line in f]

    all_predictions = [
        {'data': [json.loads(line) for line in open(PREDICTIONS_PATH_1, 'r')], 'label': 'Propella'},
        {'data': [json.loads(line) for line in open(PREDICTIONS_PATH_2, 'r')], 'label': 'Latxa-Qwen3-VL Fine-Tuned'},
        {'data': [json.loads(line) for line in open(PREDICTIONS_PATH_3, 'r')], 'label': 'Latxa-Qwen3-VL Zero-Shot'}
    ]

    raw_fields = [
        'content_type', 'business_sector', 'technical_content',
        'pii_presence', 'regional_relevance', 'country_relevance'
    ]

    fitted_encoders = {}
    gold_results = {}

    for key in configs.keys():
        encoder = get_encoder(key)
        gold_raw = [[item.get(key)] for item in gold_data]
        # Fit once on gold and store
        gold_results[f'gold_{key}'] = encoder.fit_transform(gold_raw)
        fitted_encoders[key] = encoder

    # Prepare gold data for Jaccard fields
    for field in raw_fields:
        gold_results[f'gold_{field}'] = [item.get(field) for item in gold_data]

    # --- 2. Scoring Loop ---
    # INITIALIZE BOTH DICTIONARIES HERE
    all_qwk_scores = {}
    all_jaccard_scores = {}

    for pred_set in all_predictions:
        label = pred_set['label']
        data = pred_set['data']

        current_qwk = {}
        current_jaccard = {}

        # Calculate QWK (Ordinal)
        for key in configs.keys():
            encoder = fitted_encoders[key]
            pred_raw = [[item.get(key)] for item in data]
            pred_values = encoder.transform(pred_raw)
            gold_values = gold_results[f'gold_{key}']
            current_qwk[key] = calculate_qwk(gold_values, pred_values)

        # Calculate Jaccard (Raw fields)
        for field in raw_fields:
            pred_values = [item.get(field) for item in data]
            gold_values = gold_results[f'gold_{field}']
            current_jaccard[field] = calculate_jaccard_similarity(gold_values, pred_values)

        # Store results for this specific model
        all_qwk_scores[label] = current_qwk
        all_jaccard_scores[label] = current_jaccard

    # --- 3. Plotting QWK Grouped Bar Chart ---
    labels = list(configs.keys())
    x = np.arange(len(labels))
    width = 0.2
    model_labels = list(all_qwk_scores.keys())

    plt.figure(figsize=(12, 7))
    for i, model_name in enumerate(model_labels):
        # Offset: i=0 -> x+0, i=1 -> x+0.2, i=2 -> x+0.4
        scores = [all_qwk_scores[model_name][key] for key in labels]
        plt.bar(x + (i * width), scores, width, label=model_name)

    plt.xlabel('Ordinal Property')
    plt.ylabel('QWK Score')
    plt.title('Comparison of QWK Scores across Models')
    plt.xticks(x + width, labels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('qwk_comparison_grouped.png')
    print("Saved: qwk_comparison_grouped.png")

    # --- 4. Plotting Jaccard Grouped Bar Chart ---
    labels_j = raw_fields
    x_j = np.arange(len(labels_j))

    plt.figure(figsize=(12, 7))
    for i, model_name in enumerate(model_labels):
        # Retrieve from the now-defined all_jaccard_scores
        scores = [all_jaccard_scores[model_name][field] for field in labels_j]
        plt.bar(x_j + (i * width), scores, width, label=model_name)

    plt.xlabel('Field')
    plt.ylabel('Jaccard Score')
    plt.title('Comparison of Jaccard Scores across Models')
    plt.xticks(x_j + width, labels_j, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('jaccard_comparison_grouped.png')
    print("Saved: jaccard_comparison_grouped.png")