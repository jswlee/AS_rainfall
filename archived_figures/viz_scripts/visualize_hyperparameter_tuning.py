#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results and create useful visualizations.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def plot_hyperparameter_importance(df, output_dir):
    """
    Plot hyperparameter importance based on correlation with validation loss.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial data
    output_dir : str
        Directory to save plots
    """
    # Calculate correlation between hyperparameters and validation loss
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'val_loss' and df[col].nunique() > 1]
    
    if not numeric_columns:
        print("No numeric hyperparameters found for correlation analysis")
        return
    
    # Calculate Spearman rank correlation
    importances = {}
    for col in numeric_columns:
        try:
            corr, _ = spearmanr(df[col], df['val_loss'])
            importances[col] = abs(corr)  # Use absolute value for importance
        except Exception as e:
            print(f"Error calculating correlation for {col}: {str(e)}")
    
    # Sort by importance
    importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
    
    # Print importance values
    print("\nHyperparameter Importance (based on correlation with validation loss):")
    for param, value in importances.items():
        print(f"{param}: {value:.6f}")
    
    # Plot importance
    plt.figure(figsize=(12, 8))
    params = list(importances.keys())
    values = list(importances.values())
    
    plt.barh(params, values)
    plt.xlabel('Importance (|Spearman Correlation|)')
    plt.ylabel('Hyperparameter')
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'hyperparameter_importance.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nHyperparameter importance plot saved to {output_path}")
    
    # Save as text file
    text_path = os.path.join(output_dir, 'hyperparameter_importance.txt')
    with open(text_path, 'w') as f:
        f.write("Hyperparameter Importance (based on correlation with validation loss):\n")
        for param, value in importances.items():
            f.write(f"{param}: {value:.6f}\n")
    
    print(f"Hyperparameter importance text saved to {text_path}")


def plot_hyperparameter_distributions(df, output_dir):
    """
    Plot distributions of hyperparameters.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial data
    output_dir : str
        Directory to save plots
    """
    # Select numeric hyperparameters
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'val_loss' and df[col].nunique() > 1]
    
    if not numeric_columns:
        print("No numeric hyperparameters found for distribution analysis")
        return
    
    # Create distribution plots
    print("\nCreating hyperparameter distribution plots...")
    
    # Create a directory for distribution plots
    dist_dir = os.path.join(output_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    
    # Create individual distribution plots
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        output_path = os.path.join(dist_dir, f'{col}_distribution.png')
        plt.savefig(output_path)
        plt.close()
    
    print(f"Distribution plots saved to {dist_dir}")


def plot_hyperparameter_vs_loss(df, output_dir):
    """
    Plot hyperparameters vs. validation loss.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial data
    output_dir : str
        Directory to save plots
    """
    # Select numeric hyperparameters
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'val_loss' and df[col].nunique() > 1]
    
    if not numeric_columns:
        print("No numeric hyperparameters found for scatter plot analysis")
        return
    
    # Create scatter plots
    print("\nCreating hyperparameter vs. validation loss plots...")
    
    # Create a directory for scatter plots
    scatter_dir = os.path.join(output_dir, 'scatter_plots')
    os.makedirs(scatter_dir, exist_ok=True)
    
    # Create individual scatter plots
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col, y='val_loss', data=df)
        plt.title(f'{col} vs. Validation Loss')
        plt.xlabel(col)
        plt.ylabel('Validation Loss')
        
        # Add trend line
        try:
            sns.regplot(x=col, y='val_loss', data=df, scatter=False, color='red')
        except:
            pass
        
        plt.tight_layout()
        
        output_path = os.path.join(scatter_dir, f'{col}_vs_loss.png')
        plt.savefig(output_path)
        plt.close()
    
    print(f"Scatter plots saved to {scatter_dir}")


def plot_pairwise_interactions(df, output_dir):
    """
    Plot pairwise interactions between top hyperparameters.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial data
    output_dir : str
        Directory to save plots
    """
    # Calculate correlation between hyperparameters and validation loss
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'val_loss' and df[col].nunique() > 1]
    
    if len(numeric_columns) < 2:
        print("Not enough numeric hyperparameters for pairwise interaction analysis")
        return
    
    # Calculate Spearman rank correlation
    importances = {}
    for col in numeric_columns:
        try:
            corr, _ = spearmanr(df[col], df['val_loss'])
            importances[col] = abs(corr)  # Use absolute value for importance
        except Exception as e:
            print(f"Error calculating correlation for {col}: {str(e)}")
    
    # Sort by importance and get top 5 (or fewer if less than 5 are available)
    importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
    top_params = list(importances.keys())[:min(5, len(importances))]
    
    if len(top_params) < 2:
        print("Not enough top parameters for pairwise interaction analysis")
        return
    
    # Create pairwise interaction plots
    print("\nCreating pairwise interaction plots for top hyperparameters...")
    
    # Create a directory for pairwise plots
    pairwise_dir = os.path.join(output_dir, 'pairwise_interactions')
    os.makedirs(pairwise_dir, exist_ok=True)
    
    # Create pairplot
    plt.figure(figsize=(15, 15))
    pairplot_data = df[top_params + ['val_loss']].copy()
    g = sns.pairplot(pairplot_data, corner=True, diag_kind='kde', 
                    plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                    diag_kws={'fill': True})
    g.fig.suptitle('Pairwise Interactions Between Top Hyperparameters', y=1.02, fontsize=16)
    
    output_path = os.path.join(pairwise_dir, 'top_params_pairplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of correlations
    plt.figure(figsize=(10, 8))
    corr_matrix = pairplot_data.corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Correlation Matrix of Top Hyperparameters', fontsize=14)
    
    output_path = os.path.join(pairwise_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Pairwise interaction plots saved to {pairwise_dir}")


def plot_learning_curves(df, output_dir):
    """
    Plot learning curves for best trials.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial data
    output_dir : str
        Directory to save plots
    """
    # Sort trials by validation loss
    df_sorted = df.sort_values('val_loss')
    
    # Get top 5 trials (or fewer if less than 5 are available)
    top_trials = df_sorted.head(min(5, len(df_sorted)))
    
    if len(top_trials) == 0:
        print("No trials available for learning curve analysis")
        return
    
    print("\nAnalyzing top trials for learning curves...")
    
    # Create a directory for learning curve plots
    curves_dir = os.path.join(output_dir, 'learning_curves')
    os.makedirs(curves_dir, exist_ok=True)
    
    # Create a summary of top trials
    summary_path = os.path.join(curves_dir, 'top_trials_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Top Trials Summary:\n\n")
        for i, (_, trial) in enumerate(top_trials.iterrows()):
            f.write(f"Rank {i+1} (Trial {trial['trial_id']}):\n")
            f.write(f"  Validation Loss: {trial['val_loss']:.6f}\n")
            f.write("  Hyperparameters:\n")
            for col in df.columns:
                if col not in ['trial_id', 'val_loss']:
                    f.write(f"    {col}: {trial[col]}\n")
            f.write("\n")
    
    print(f"Top trials summary saved to {summary_path}")
    
    # Create a bar chart of top trials
    plt.figure(figsize=(12, 6))
    trial_ids = [f"Trial {trial['trial_id']}" for _, trial in top_trials.iterrows()]
    val_losses = [trial['val_loss'] for _, trial in top_trials.iterrows()]
    
    bars = plt.bar(trial_ids, val_losses)
    plt.xlabel('Trial')
    plt.ylabel('Validation Loss')
    plt.title('Top Performing Trials')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = os.path.join(curves_dir, 'top_trials_comparison.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Top trials comparison plot saved to {output_path}")


def main():
    """
    Main function to analyze hyperparameter tuning results.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--tuner_dir', type=str, required=True,
                        help='Directory containing the tuner results')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory to save the plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find trial directories
    project_dir = os.path.join(args.tuner_dir, 'land_model_extended_tuning')
    if not os.path.exists(project_dir):
        print(f"Project directory not found: {project_dir}")
        print("Looking for trial files directly in the tuner directory...")
        project_dir = args.tuner_dir
    
    trial_dirs = glob.glob(os.path.join(project_dir, 'trial_*'))
    if not trial_dirs:
        print(f"No trial directories found in {project_dir}")
        return 1
    
    print(f"Found {len(trial_dirs)} trial directories")
    
    # Collect trial data
    trials_data = []
    for trial_dir in trial_dirs:
        trial_id = os.path.basename(trial_dir)
        
        # Try to find the trial.json file
        trial_json = os.path.join(trial_dir, 'trial.json')
        if not os.path.exists(trial_json):
            print(f"Trial JSON not found: {trial_json}")
            continue
        
        try:
            with open(trial_json, 'r') as f:
                trial_data = json.load(f)
            
            # Extract hyperparameters and score
            hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
            metrics = trial_data.get('metrics', {}).get('metrics', {})
            
            # Get the best validation loss
            val_loss = None
            if 'val_loss' in metrics:
                val_loss_data = metrics['val_loss']
                if 'observations' in val_loss_data and val_loss_data['observations']:
                    val_loss = val_loss_data['observations'][0]['value'][0]
            
            if val_loss is not None:
                # Create a record with hyperparameters and score
                record = {'trial_id': trial_id, 'val_loss': val_loss}
                record.update(hyperparams)
                trials_data.append(record)
        except Exception as e:
            print(f"Error processing trial {trial_id}: {str(e)}")
    
    if not trials_data:
        print("No valid trial data found")
        return 1
    
    # Convert to DataFrame
    df = pd.DataFrame(trials_data)
    print(f"Collected data for {len(df)} trials")
    
    # Create various plots
    plot_hyperparameter_importance(df, args.output_dir)
    plot_hyperparameter_distributions(df, args.output_dir)
    plot_hyperparameter_vs_loss(df, args.output_dir)
    plot_pairwise_interactions(df, args.output_dir)
    plot_learning_curves(df, args.output_dir)
    
    # Analyze the best trial
    best_trial = df.loc[df['val_loss'].idxmin()]
    print("\nBest Trial:")
    for col in df.columns:
        if col != 'trial_id':
            print(f"{col}: {best_trial[col]}")
    
    # Save best hyperparameters
    best_path = os.path.join(args.output_dir, 'best_hyperparameters_analyzed.txt')
    with open(best_path, 'w') as f:
        f.write("Best hyperparameters from analysis:\n")
        for col in df.columns:
            if col not in ['trial_id', 'val_loss']:
                f.write(f"{col}: {best_trial[col]}\n")
    
    print(f"\nBest hyperparameters saved to {best_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
