import argparse
import json
import os
import traceback
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.evaluation import evaluate_gene_selection


def average_metrics(metrics_list):
    """Average a list of metric dictionaries."""
    if not metrics_list:
        return {}

    avg_metrics = {}
    for metric in metrics_list[0]:
        avg_metrics[metric] = float(np.round(np.nanmean([p[metric] for p in metrics_list]), 2))

    return avg_metrics


def evaluate_dataset_selection(pred_dir, ref_dir):
    """Placeholder for dataset selection evaluation."""
    print("Dataset selection evaluation is not implemented yet.")
    return {"placeholder": "Dataset selection evaluation not implemented"}


def calculate_jaccard(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 0.0 if union == 0 else intersection / union


def calculate_pearson_correlation(df1, df2):
    """Calculate Pearson correlation between common features in two dataframes."""
    common_samples = df1.index.intersection(df2.index)
    common_features = df1.columns.intersection(df2.columns)
    
    if len(common_samples) == 0 or len(common_features) == 0:
        return 0.0
    
    aligned_df1 = df1.loc[common_samples, common_features]
    aligned_df2 = df2.loc[common_samples, common_features]
    
    correlations = []
    for col in common_features:
        # Handle missing values based on column type
        s1 = aligned_df1[col].copy()
        s2 = aligned_df2[col].copy()
        
        # Fill NAs: mode for Gender, mean for others
        if col == 'Gender':
            # Handle empty series or all-NA series by using 0.0
            s1_mode = s1.mode().iloc[0] if not s1.empty and s1.notna().any() else 0.0
            s2_mode = s2.mode().iloc[0] if not s2.empty and s2.notna().any() else 0.0
            s1.fillna(s1_mode, inplace=True)
            s2.fillna(s2_mode, inplace=True)
        else:
            # Handle empty series or all-NA series by using 0.0
            s1_mean = s1.mean() if not s1.empty and s1.notna().any() else 0.0
            s2_mean = s2.mean() if not s2.empty and s2.notna().any() else 0.0
            s1.fillna(s1_mean, inplace=True)
            s2.fillna(s2_mean, inplace=True)
        
        # Skip if not enough data points or handle constant vectors
        if len(s1) <= 1 or len(s2) <= 1:
            correlations.append(0)
            continue
            
        std1 = np.std(s1)
        std2 = np.std(s2)
        if std1 == 0 and std2 == 0:
            correlations.append(1.0)
        elif std1 == 0 or std2 == 0:
            correlations.append(0.0)
        else:
            correlations.append(np.corrcoef(s1, s2)[0, 1])

    return 0.0 if len(correlations) == 0 else np.nanmean(correlations)


def evaluate_csv(pred_file_path, ref_file_path, subtask="linked"):
    """
    Evaluate preprocessing by comparing prediction and reference CSV files.
    
    Args:
        pred_file_path: Path to the prediction CSV file
        ref_file_path: Path to the reference CSV file
        subtask: The preprocessing subtask ('gene', 'clinical', 'linked')
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Default metrics if file doesn't exist
    default_metrics = {
        'attributes_jaccard': 0.0,
        'samples_jaccard': 0.0,
        'feature_correlation': 0.0,
        'composite_similarity_correlation': 0.0
    }
    
    # Check if prediction file exists
    if not os.path.isfile(pred_file_path):
        return default_metrics
    
    try:
        # Read CSV files
        df1 = pd.read_csv(pred_file_path, index_col=0)
        df2 = pd.read_csv(ref_file_path, index_col=0)
        
        # Reset index and column names to avoid possible errors and confusion
        df1.index.name = None
        df1.columns.name = None
        df2.index.name = None
        df2.columns.name = None
        
        # Make sure rows represent samples and columns represent features
        if subtask != "linked":
            # Transpose the DataFrames
            df1 = df1.T
            df2 = df2.T
        
        # Return default metrics if any dataframe is empty
        if df1.empty or df2.empty:
            return default_metrics

        # Calculate metrics
        attributes_jaccard = calculate_jaccard(set(df1.columns), set(df2.columns))
        samples_jaccard = calculate_jaccard(set(df1.index), set(df2.index))
        feature_correlation = calculate_pearson_correlation(df1, df2)
        composite_similarity_correlation = attributes_jaccard * samples_jaccard * feature_correlation

        return {
            'attributes_jaccard': attributes_jaccard,
            'samples_jaccard': samples_jaccard,
            'feature_correlation': feature_correlation,
            'composite_similarity_correlation': composite_similarity_correlation
        }
    except Exception as e:
        print(f"Error processing {pred_file_path} and {ref_file_path}")
        print(f"Error details: {str(e)}")
        print(traceback.format_exc())
        return default_metrics


def evaluate_dataset_preprocessing(pred_dir, ref_dir, subtasks=None):
    """
    Evaluate preprocessing by comparing predicted and reference datasets.
    
    Args:
        pred_dir: Path to prediction directory
        ref_dir: Path to reference directory
        subtasks: List of subtasks to evaluate ('gene', 'clinical', 'linked')
                 or None to evaluate all
    
    Returns:
        Dictionary of evaluation metrics for each subtask
    """
    results = {}
    if subtasks is None:
        subtasks = ["gene", "clinical", "linked"]
    
    pred_preprocess_dir = os.path.join(pred_dir, "preprocess")
    ref_preprocess_dir = os.path.join(ref_dir, "preprocess")
    
    if not os.path.exists(pred_preprocess_dir):
        print(f"Warning: Preprocessing prediction directory '{pred_preprocess_dir}' does not exist.")
        return {subtask: {} for subtask in subtasks}
    
    for subtask in subtasks:
        metrics = []
        
        # Get list of trait directories
        trait_dirs = []
        for t in os.listdir(ref_preprocess_dir):
            ref_trait_dir = os.path.join(ref_preprocess_dir, t)
            if os.path.isdir(ref_trait_dir):
                trait_dirs.append(t)
        
        # Process each trait directory with progress bar
        with tqdm(total=len(trait_dirs), desc=f"Evaluating {subtask} data preprocessing") as pbar:
            for trait in trait_dirs:
                ref_trait_dir = os.path.join(ref_preprocess_dir, trait)
                
                # Determine the subdirectory path based on subtask
                if subtask in ["gene", "clinical"]:
                    sub_dir = os.path.join(ref_trait_dir, f"{subtask}_data")
                else:  # linked
                    sub_dir = ref_trait_dir
                
                if not os.path.isdir(sub_dir):
                    pbar.update(1)
                    continue
                
                # Process each CSV file (without nested progress bar)
                for file in os.listdir(sub_dir):
                    if file.endswith(".csv"):
                        ref_file_path = os.path.join(sub_dir, file)
                        
                        # Get corresponding prediction file path
                        if subtask in ["gene", "clinical"]:
                            pred_file_path = os.path.join(pred_preprocess_dir, trait, f"{subtask}_data", file)
                        else:  # linked
                            pred_file_path = os.path.join(pred_preprocess_dir, trait, file)
                        
                        # Evaluate the file pair
                        rs = evaluate_csv(pred_file_path, ref_file_path, subtask)
                        metrics.append(rs)
                
                pbar.update(1)
        
        results[subtask] = average_metrics(metrics)
    
    return results


def evaluate_problem_result(ref_file, pred_file):
    """Calculate metrics for gene selection evaluation."""
    assert os.path.exists(ref_file), "Reference file does not exist"
    with open(ref_file, 'r') as rfile:
        ref = json.load(rfile)
    ref_genes = ref["significant_genes"]["Variable"]

    # If the 'pred_file' does not exist, it indicates the agent's regression code fails to run on this question
    metrics = {'success': 0.0,
               'precision': np.nan,
               'recall': np.nan,
               'f1': np.nan,
               'auroc': np.nan,
               'gsea_es': np.nan,
               'trait_pred_accuracy': np.nan,
               'trait_pred_f1': np.nan}

    if os.path.exists(pred_file):
        with open(pred_file, 'r') as file:
            result = json.load(file)
        pred_genes = result["significant_genes"]["Variable"]
        metrics.update(evaluate_gene_selection(pred_genes, ref_genes))

        # Optionally, record performance on trait prediction.
        try:
            metrics['trait_pred_accuracy'] = result["cv_performance"]["prediction"]["accuracy"]
        except KeyError:
            pass
        try:
            metrics['trait_pred_f1'] = result["cv_performance"]["prediction"]["f1"]
        except KeyError:
            pass

        metrics['success'] = 100.0

    return metrics


def categorize_and_aggregate(results):
    """Categorize and aggregate metrics by condition type."""
    categorized_results = {'Unconditional one-step': [], 'Conditional one-step': [], 'Two-step': []}
    for pair, metrics in results.items():
        condition = pair[1]
        if condition is None or condition.lower() == "none":
            category = 'Unconditional one-step'
        elif condition.lower() in ["age", "gender"]:
            category = 'Conditional one-step'
        else:
            category = 'Two-step'
        categorized_results[category].append(metrics)

    aggregated_metrics = {}
    for category, metrics_list in categorized_results.items():
        aggregated_metrics[category] = average_metrics(metrics_list)
    aggregated_metrics['Overall'] = average_metrics(
        [metric for sublist in categorized_results.values() for metric in sublist])
    return aggregated_metrics


def evaluate_statistical_analysis(pred_dir, ref_dir):
    """Evaluate statistical analysis (gene selection) task."""
    results = {}
    pred_regress_dir = os.path.join(pred_dir, 'regress')
    ref_regress_dir = os.path.join(ref_dir, 'regress')
    
    if not os.path.exists(pred_regress_dir):
        print(f"Warning: Statistical analysis prediction directory '{pred_regress_dir}' does not exist.")
        return {}, {}
    
    trait_dirs = [t for t in sorted(os.listdir(ref_regress_dir)) 
                 if os.path.isdir(os.path.join(ref_regress_dir, t))]
    
    with tqdm(total=len(trait_dirs), desc="Evaluating statistical analysis") as pbar:
        for trait in trait_dirs:
            ref_trait_path = os.path.join(ref_regress_dir, trait)
            
            # Process each JSON file (without nested progress bar)
            json_files = [f for f in sorted(os.listdir(ref_trait_path)) 
                         if f.startswith('significant_genes') and f.endswith('.json')]
            
            for filename in json_files:
                parts = filename.split('_')
                condition = '_'.join(parts[3:])[:-5]
                ref_file = os.path.join(ref_trait_path, filename)
                pred_file = os.path.join(pred_regress_dir, trait, filename)
                
                try:
                    metrics = evaluate_problem_result(ref_file, pred_file)
                    results[(trait, condition)] = metrics
                except Exception as e:
                    print(f"Error evaluating {pred_file}")
                    print(f"Error details: {str(e)}")
                    print(traceback.format_exc())
            
            pbar.update(1)
    
    categorized_avg_metrics = categorize_and_aggregate(results)
    return results, categorized_avg_metrics


def main(pred_dir, ref_dir, tasks=None, preprocess_subtasks=None):
    """
    Main evaluation function that can evaluate different tasks.
    
    Args:
        pred_dir: Path to prediction directory
        ref_dir: Path to reference directory
        tasks: List of tasks to evaluate ('selection', 'preprocessing', 'analysis')
               or None to evaluate all
        preprocess_subtasks: List of preprocessing subtasks to evaluate
                           ('gene', 'clinical', 'linked') or None to evaluate all
    
    Returns:
        Dictionary of evaluation results for each task
    """
    if tasks is None:
        tasks = ["selection", "preprocessing", "analysis"]
    
    results = {}
    
    # Evaluate dataset selection
    if "selection" in tasks:
        print("\n=== Evaluating Dataset Selection ===")
        results["selection"] = evaluate_dataset_selection(pred_dir, ref_dir)
    
    # Evaluate preprocessing
    if "preprocessing" in tasks:
        print("\n=== Evaluating Dataset Preprocessing ===")
        results["preprocessing"] = evaluate_dataset_preprocessing(pred_dir, ref_dir, preprocess_subtasks)
    
    # Evaluate statistical analysis
    if "analysis" in tasks:
        print("\n=== Evaluating Statistical Analysis ===")
        analysis_results, categorized_metrics = evaluate_statistical_analysis(pred_dir, ref_dir)
        results["analysis"] = {
            "detailed_results": analysis_results,
            "categorized_metrics": categorized_metrics
        }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM agent-based methods for gene expression data analysis")
    parser.add_argument("-p", "--pred-dir", type=str, required=True, 
                      help="Path to the prediction directory")
    parser.add_argument("-r", "--ref-dir", type=str, default="./output", 
                      help="Path to the reference directory")
    parser.add_argument("-t", "--tasks", type=str, nargs="+", 
                      choices=["selection", "preprocessing", "analysis"], default=None,
                      help="Tasks to evaluate (default: all)")
    parser.add_argument("-s", "--preprocess-subtasks", type=str, nargs="+", 
                      choices=["gene", "clinical", "linked"], default=None,
                      help="Preprocessing subtasks to evaluate (default: all)")

    args = parser.parse_args()

    try:
        results = main(args.pred_dir, args.ref_dir, args.tasks, args.preprocess_subtasks)
        
        print("\n=== Evaluation Results ===")
        
        # Print dataset selection results if available
        if "selection" in results:
            print("\nDataset Selection Results:")
            print(results["selection"])
        
        # Print preprocessing results if available
        if "preprocessing" in results:
            print("\nPreprocessing Results:")
            for subtask, metrics in results["preprocessing"].items():
                print(f"\n{subtask.capitalize()} preprocessing:")
                if metrics:
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
                else:
                    print("  No results available")
        
        # Print statistical analysis results if available
        if "analysis" in results:
            print("\nStatistical Analysis Results:")
            categorized_metrics = results["analysis"]["categorized_metrics"]
            for category, metrics in categorized_metrics.items():
                print(f"\n{category}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.2f}")
    
    except Exception as e:
        print(f"Error in evaluation process: {str(e)}")
        print(traceback.format_exc())
