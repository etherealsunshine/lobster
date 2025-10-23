#!/usr/bin/env python3
"""
WandB-Integrated Generation Script for genUME Parameter Optimization

This script integrates with wandb sweeps to optimize generation parameters.
Based on the wandb sweeps walkthrough: https://docs.wandb.ai/guides/sweeps/walkthrough/

Usage:
    wandb sweep <path to yaml>
    Update the id in the wandb_slurm.sh script to the wandb run id:
    srun -u --cpus-per-task $SLURM_CPUS_PER_TASK --cpu-bind=cores,verbose wandb agent prescient-design/lobster-wandb_sweeps/<wandb run id>
    sbatch wandb_slurm.sh
"""

from pathlib import Path

from loguru import logger
import pandas as pd

import wandb
from omegaconf import DictConfig, OmegaConf

# Import the original generation function
from lobster.cmdline.generate import generate as run_generation


def objective(config):
    """
    Objective function for wandb sweep optimization.

    Args:
        config: wandb config object containing sweep parameters

    Returns:
        float: Composite score for optimization
    """
    # Create config from wandb parameters
    gen_config = create_config_from_wandb(config)

    # Run generation
    run_generation(gen_config)

    # Collect metrics
    metrics = collect_metrics_from_output(gen_config.output_dir)

    # Calculate composite score
    composite_score = calculate_composite_score(metrics)

    # Log metrics to wandb
    wandb.log({**metrics, "composite_score": composite_score})

    logger.info(f"Run completed with composite score: {composite_score:.4f}")
    return composite_score


def create_config_from_wandb(config) -> DictConfig:
    """Create genUME config from wandb sweep parameters."""

    # Get generation mode from config
    mode = config.get("mode", "unconditional")

    # Base config structure
    config_dict = {
        "output_dir": f"./wandb_outputs/{wandb.run.id}",
        "seed": 12345,
        "model": {
            "_target_": "lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule",
            "ckpt_path": "/data2/ume/gen_ume/runs//2025-10-08T23-54-39/last.ckpt",
        },
        "generation": {
            "mode": mode,
            "length": config.get("length", 200),
            "num_samples": config.get("num_samples", 10),
            "nsteps": config.get("nsteps", 200),
            "temperature_seq": config.get("temperature_seq", 0.5),
            "temperature_struc": config.get("temperature_struc", 0.5),
            "stochasticity_seq": config.get("stochasticity_seq", 20),
            "stochasticity_struc": config.get("stochasticity_struc", 20),
            "use_esmfold": True,
            "max_length": 512,
            "save_csv_metrics": True,
            "create_plots": False,
            "batch_size": config.get("batch_size", 1),
            "n_trials": config.get("n_trials", 1),
            "input_structures": config.get("input_structures", None),
        },
    }

    # Mode-specific adjustments
    if mode == "inverse_folding":
        # Inverse folding specific settings
        if config_dict["generation"]["input_structures"] is None:
            raise ValueError("input_structures must be provided for inverse folding mode")
        logger.info(f"Using input structures: {config_dict['generation']['input_structures']}")

    elif mode == "forward_folding":
        # Forward folding specific settings
        if config_dict["generation"]["input_structures"] is None:
            raise ValueError("input_structures must be provided for forward folding mode")
        logger.info(f"Using input structures: {config_dict['generation']['input_structures']}")

    elif mode == "inpainting":
        # Inpainting specific settings
        if config_dict["generation"]["input_structures"] is None:
            raise ValueError("input_structures must be provided for inpainting mode")

        # Add inpainting-specific parameters
        config_dict["generation"]["mask_indices_sequence"] = config.get("mask_indices_sequence", "")
        config_dict["generation"]["mask_indices_structure"] = config.get("mask_indices_structure", "")

        logger.info(f"Using input structures: {config_dict['generation']['input_structures']}")
        logger.info(f"Sequence mask indices: {config_dict['generation']['mask_indices_sequence']}")
        logger.info(f"Structure mask indices: {config_dict['generation']['mask_indices_structure']}")

    elif mode == "unconditional":
        # Unconditional generation - no input structures needed
        config_dict["generation"]["input_structures"] = None

    return OmegaConf.create(config_dict)


def collect_metrics_from_output(output_dir: str) -> dict[str, float]:
    """Collect metrics from generation output CSV files."""
    output_path = Path(output_dir)
    metrics = {}

    # Look for metrics CSV files
    csv_files = list(output_path.glob("*_metrics_*.csv"))

    if not csv_files:
        logger.warning(f"No metrics CSV files found in {output_dir}")
        return metrics

    # Use the most recent CSV file
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)

    df = pd.read_csv(latest_csv)

    # Define metrics based on generation mode
    mode = df["mode"].iloc[0] if "mode" in df.columns and len(df) > 0 else "unconditional"

    if mode == "unconditional":
        metric_columns = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]
    elif mode == "inverse_folding":
        metric_columns = ["percent_identity", "plddt", "predicted_aligned_error", "tm_score", "rmsd"]
    elif mode == "forward_folding":
        metric_columns = ["tm_score", "rmsd"]
    elif mode == "inpainting":
        metric_columns = [
            "percent_identity_masked",
            "percent_identity_unmasked",
            "rmsd_inpainted",
            "plddt",
            "predicted_aligned_error",
            "tm_score",
            "rmsd",
        ]
    else:
        metric_columns = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]

    for metric in metric_columns:
        if metric in df.columns:
            # Convert to numeric and remove NaN values
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(values) > 0:
                metrics[f"avg_{metric}"] = float(values.mean())
                metrics[f"std_{metric}"] = float(values.std())
                metrics[f"min_{metric}"] = float(values.min())
                metrics[f"max_{metric}"] = float(values.max())
                metrics[f"count_{metric}"] = len(values)

    # Calculate additional metrics
    if "sequence_length" in df.columns:
        lengths = pd.to_numeric(df["sequence_length"], errors="coerce").dropna()
        if len(lengths) > 0:
            metrics["avg_sequence_length"] = float(lengths.mean())
            metrics["std_sequence_length"] = float(lengths.std())

    logger.info(f"Collected metrics for {mode} mode: {metrics}")

    return metrics


def calculate_composite_score(metrics: dict[str, float]) -> float:
    """
    Calculate composite score for optimization.

    Different scoring strategies for different modes:
    - Unconditional: Focus on plddt, tm_score, minimize PAE and RMSD
    - Inverse folding: Focus on percent_identity, plddt, tm_score
    - Forward folding: Focus on tm_score, minimize RMSD
    - Inpainting: Focus on percent_identity_masked, rmsd_inpainted (post-alignment), tm_score, plddt
    """
    score = 0.0

    # Determine mode from available metrics
    mode = "unconditional"  # default
    if "avg_percent_identity_masked" in metrics or "avg_rmsd_inpainted" in metrics:
        mode = "inpainting"
    elif "avg_percent_identity" in metrics:
        mode = "inverse_folding"
    elif "avg_tm_score" in metrics and "avg_plddt" not in metrics:
        mode = "forward_folding"

    if mode == "unconditional":
        # Higher is better: plddt, tm_score
        if "avg_plddt" in metrics:
            score += metrics["avg_plddt"] * 0.3

        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.3

        # Lower is better: predicted_aligned_error, rmsd
        if "avg_predicted_aligned_error" in metrics:
            score -= metrics["avg_predicted_aligned_error"] / 100 * 0.2

        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.2

    elif mode == "inverse_folding":
        # Higher is better: percent_identity, plddt, tm_score
        if "avg_percent_identity" in metrics:
            score += metrics["avg_percent_identity"] * 0.4  # Most important for inverse folding

        if "avg_plddt" in metrics:
            score += metrics["avg_plddt"] * 0.2

        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.4

        # Lower is better: predicted_aligned_error, rmsd
        if "avg_predicted_aligned_error" in metrics:
            score -= metrics["avg_predicted_aligned_error"] / 100 * 0.1

        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.1

    elif mode == "forward_folding":
        # Higher is better: tm_score
        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.7  # Most important for forward folding

        # Lower is better: rmsd
        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.3

    elif mode == "inpainting":
        # Higher is better: percent_identity_masked, percent_identity_unmasked, plddt, tm_score
        if "avg_percent_identity_masked" in metrics:
            score += metrics["avg_percent_identity_masked"] * 0.25  # How well we recovered masked sequence

        if "avg_percent_identity_unmasked" in metrics:
            score += metrics["avg_percent_identity_unmasked"] * 0.15  # Preserved unmasked regions

        if "avg_plddt" in metrics:
            score += metrics["avg_plddt"] * 0.0015  # Structure quality (scale to 0-100 range)

        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.20  # Overall structural similarity

        # Lower is better: rmsd_inpainted, predicted_aligned_error, rmsd
        if "avg_rmsd_inpainted" in metrics:
            score -= metrics["avg_rmsd_inpainted"] / 5 * 0.20  # KEY: minimize structural deviation in inpainted region

        if "avg_predicted_aligned_error" in metrics:
            score -= metrics["avg_predicted_aligned_error"] / 100 * 0.05  # Minimize PAE

        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.05  # Minimize overall RMSD

    logger.info(f"Calculated composite score for {mode} mode: {score:.4f}")
    return score


def main():
    """
    Main function for wandb-integrated generation.
    Based on the wandb sweeps walkthrough pattern.
    """
    # Initialize wandb run
    with wandb.init(project="genume-parameter-optimization") as run:
        # Run objective function
        score = objective(run.config)

        # Log final score
        wandb.log({"final_score": score})

        logger.info(f"WandB run completed successfully with score: {score:.4f}")


if __name__ == "__main__":
    main()
