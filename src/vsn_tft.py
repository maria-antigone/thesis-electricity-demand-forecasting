"""
This script extracts and visualizes the Variable Selection Network (VSN) 
feature importances from a pre-trained Temporal Fusion Transformer (TFT) model.

It performs a robust analysis by AVERAGING scores over several batches from the test set.
It saves the importance plots as PNG files and the importance scores as CSV files.

Usage (from project root):
    python src/vsn_tft.py --horizon short
"""
import os
import yaml
import argparse
import pandas as pd
import glob
import torch
import warnings

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Import necessary functions from the existing pipeline
from data_processing_tft import add_time_idx_and_series_id, split_dataset, create_tft_dataset

# Silence repetitive warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*and is already saved*")


def find_best_model_path(run_dir: str) -> str:
    """Finds the best model checkpoint file in a given run directory."""
    checkpoint_files = glob.glob(os.path.join(run_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (.ckpt) found in the directory: {run_dir}")
    
    print(f"Found model checkpoint: {checkpoint_files[0]}")
    return checkpoint_files[0]

def extract_and_plot_vsn(horizon: str):
    """
    Loads a pre-trained TFT model, extracts VSN weights, and plots feature importance.
    
    Args:
        horizon (str): The forecasting horizon ('short', 'medium', 'long') to analyze.
    """
    print(f"--- Starting VSN Extraction for {horizon.upper()} Horizon ---", flush=True)

    # 1. Load Configuration and Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    config_file_path = os.path.join(project_base_dir, "src", "config.yaml")

    with open(config_file_path) as file:
        full_config = yaml.safe_load(file)
    
    horizon_config = full_config[horizon]
    tft_config = horizon_config['tft']
    print("Configuration loaded successfully.")

    run_tag = tft_config.get('run_tag', f"tft_{horizon}")
    run_name = f"{run_tag}_h{horizon_config['output_horizon']}_w{horizon_config['input_window']}"
    run_output_dir = os.path.join(project_base_dir, "outputs", "tft_runs", run_name)
    
    if not os.path.exists(run_output_dir):
        raise FileNotFoundError(f"Run directory not found: {run_output_dir}.\nPlease run training first.")
    print(f"Located run directory: {run_output_dir}")

    # 2. Load Model and Data
    model_path = find_best_model_path(run_output_dir)
    model = TemporalFusionTransformer.load_from_checkpoint(model_path).cpu()
    print("TFT model loaded successfully.")

    data_file_name = "merged_dataset_featurized.csv"
    data_path = os.path.join(project_base_dir, "data", "processed", data_file_name)
    df = pd.read_csv(data_path, sep=";", parse_dates=["utc_timestamp"])
    df = add_time_idx_and_series_id(df)

    train_df, _, test_df = split_dataset(df, horizon_config)
    
    print("Creating datasets to build test dataloader...")
    # The training_dataset object is the single source of truth for feature names
    training_dataset = create_tft_dataset(train_df, horizon_config, horizon_config["target_column"])
    testing_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, stop_randomization=True)

    test_loader = testing_dataset.to_dataloader(
        train=False, 
        batch_size=tft_config.get("batch_size", 16),
        num_workers=0
    )
    print("Test dataloader created successfully.")

    # --- START OF REVISED SECTION FOR ROBUST ANALYSIS ---
    # 3. Calculate Average Importance over SEVERAL BATCHES
    print("\nCalculating feature importance by averaging over several batches...")
    n_batches_to_average = 10 
    interpretation_sum = {}

    # Loop over a few batches
    for i, (x, y) in enumerate(test_loader):
        if i >= n_batches_to_average:
            break
        
        # We must allow gradients for interpretation
        raw_predictions = model(x)
        
        # Sum the importances from this batch
        interpretation_batch = model.interpret_output(raw_predictions, reduction="sum")
        
        # Aggregate the sums
        for key, value in interpretation_batch.items():
            if key in interpretation_sum:
                interpretation_sum[key] += value
            else:
                interpretation_sum[key] = value.clone()

    # Average the importances by dividing by the number of batches
    interpretation = {key: value / n_batches_to_average for key, value in interpretation_sum.items()}

    print("Finished calculating averaged importances.")
    # --- END OF REVISED SECTION ---
    
    print(f"Interpretation data keys found: {list(interpretation.keys())}")

    # 4. Plot and Save Visualizations (PNGs)
    print("\nGenerating and saving feature importance plots...")
    figs_dict = model.plot_interpretation(interpretation)
    
    plot_keys_to_save = ['encoder_variables', 'decoder_variables', 'static_variables']

    for key in plot_keys_to_save:
        if key in figs_dict:
            fig = figs_dict[key]
            plot_title = key.replace('_', ' ').title()
            fig.suptitle(f"TFT Importance: {plot_title}\n({horizon.capitalize()} Horizon, Averaged)", fontsize=16)
            plot_save_path = os.path.join(run_output_dir, f"vsn_importance_{key}_{horizon}.png")
            fig.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved plot to: {plot_save_path}")

    # 5. Save Importance Scores (CSVs)
    print("\nSaving feature importance scores to CSV files...")

    # Define a mapping from interpretation keys to the dataset's feature name lists
    # This is the single source of truth for names.
    key_to_names_map = {
        "encoder_variables": training_dataset.time_varying_known_reals + training_dataset.time_varying_unknown_reals,
        "decoder_variables": training_dataset.time_varying_known_reals,
        "static_variables": training_dataset.static_reals
    }

    for type_key, feature_names in key_to_names_map.items():
        if type_key in interpretation:
            scores = interpretation[type_key]
            
            # Ensure we have a tensor with scores and a corresponding list of names
            if isinstance(scores, torch.Tensor) and scores.numel() > 0 and len(feature_names) == scores.numel():
                df_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': scores.detach().cpu().numpy().flatten()
                })
                df_importance = df_importance.sort_values('importance', ascending=False).reset_index(drop=True)
                
                csv_save_path = os.path.join(run_output_dir, f"vsn_scores_{type_key}_{horizon}.csv")
                df_importance.to_csv(csv_save_path, index=False)
                print(f"  - Saved scores to: {csv_save_path}")

                print(f"\nImportance for '{type_key}':")
                print(df_importance)
            else:
                 print(f"\nCould not save scores for '{type_key}'. Data format or length mismatch.")
            
    print(f"\n--- VSN Extraction for {horizon.upper()} Horizon Finished ---\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and visualize Variable Selection Network (VSN) feature importances from a trained TFT model."
    )
    parser.add_argument(
        "--horizon", 
        type=str, 
        default="short", 
        choices=["short", "medium", "long"],
        help="The forecasting horizon of the trained model to analyze (default: short)."
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    extract_and_plot_vsn(horizon=args.horizon)