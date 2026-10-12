# Physics-Informed Patch Transformer with RevIN for Zero-Shot Cross-Station Time-Series Forecasting

A research repository for an M.Tech thesis investigating **zero-shot, cross-station time-series forecasting** under distribution shift. The project studies transfer from **ETTm1** to **ETTm2**, with emphasis on out-of-distribution (OOD) generalization, non-stationary operating conditions, and physically plausible long-horizon predictions.

## Project Overview

Time-series forecasting models often perform well when training and test data are drawn from the same station or operating regime. However, performance can degrade substantially when a model is transferred to a previously unseen station whose marginal statistics, scale, variance, and temporal dynamics differ from the source domain.

This repository addresses that challenge through a **Physics-Informed Patch Transformer with Reversible Instance Normalization (RevIN)**. The model is designed for a zero-shot setting:

- **Source domain:** ETTm1
- **Target domain:** ETTm2
- **Target data usage:** No target-domain fine-tuning during the zero-shot transfer evaluation
- **Primary concern:** Robustness to distribution shift and non-stationary temporal statistics

The approach combines data-adaptive normalization, a patch-based Transformer backbone, and a thermodynamic regularization objective. The resulting system aims to improve predictive accuracy while discouraging implausible temporal drift in the forecast trajectory.

## Research Objectives

1. Evaluate whether reversible, instance-level normalization improves cross-station transfer.
2. Quantify the effect of a physics-informed thermodynamic loss on forecast behavior.
3. Compare a patch-based Transformer representation against conventional point-by-point sequence modeling assumptions.
4. Measure the interaction between statistical normalization and physical regularization under a strict deterministic protocol.

## Core Architecture

### 1. Reversible Instance Normalization (RevIN)

RevIN dynamically normalizes each input instance using its own temporal statistics. This reduces the impact of non-stationary shifts in level and scale between stations or operating regimes.

The normalization process is reversible:

1. Compute instance-specific location and scale statistics from the input window.
2. Normalize the input before it is passed to the forecasting backbone.
3. Predict in the normalized space.
4. Restore the original thermodynamic scale through denormalization before computing the final physical forecast.

This design allows the model to learn transferable temporal structure while preserving the target instance's physical scale during reconstruction and evaluation.

### 2. Physics-Informed Thermodynamic Loss

The training objective includes a thermodynamic regularization term that penalizes **unphysical temporal drift**. In addition to minimizing forecasting error, the loss constrains the predicted trajectory to avoid implausible changes in thermodynamic variables over time.

A general form of the objective is:

```text
L_total = L_forecast + lambda_physics * L_thermodynamic
```

where:

- `L_forecast` is the supervised forecasting loss, such as mean squared error.
- `L_thermodynamic` penalizes physically inconsistent temporal behavior.
- `lambda_physics` controls the strength of the physical regularization.

The physics term is intended as a soft constraint rather than a replacement for data-driven forecasting. Its role is to improve plausibility and stability, particularly over long horizons and under distribution shift.

### 3. Patch-Based Transformer Backbone

The model divides a long input sequence into temporal patches and embeds each patch before applying Transformer-based sequence modeling. Patch-level tokenization offers several advantages for long-horizon forecasting:

- Reduces the effective sequence length presented to self-attention.
- Captures local temporal patterns as higher-level representations.
- Improves computational efficiency relative to point-by-point attention.
- Provides a natural interface for modeling multi-scale temporal structure.

This contrasts with traditional point-by-point LSTM pipelines, which process every time step sequentially and may face optimization and memory limitations when modeling long contexts.

## End-to-End Pipeline

```text
Raw station window
        |
        v
RevIN instance normalization
        |
        v
Temporal patching and embedding
        |
        v
Transformer encoder / forecasting head
        |
        v
Reversible thermodynamic-scale denormalization
        |
        v
Forecast reconstruction
        |
        +--> Forecasting loss
        +--> Thermodynamic physics loss
```

## Empirical Ablation Results

The following results come from a strict deterministic evaluation protocol:

- Random seed: `42`
- Training duration: `30` epochs
- Transfer setting: ETTm1 → ETTm2
- Metric: Mean squared error (MSE)

| Configuration | RevIN | Physics-Informed Loss | MSE |
|:--|:--:|:--:|--:|
| Baseline | No | No | **2.8369** |
| Variant A | Yes | No | **0.5994** |
| Variant B | No | Yes | **3.9302** |
| Full Model | Yes | Yes | **0.5801** |

### Interpretation

The ablation matrix indicates that RevIN is the dominant contributor to cross-station transfer performance in this evaluation, reducing MSE from `2.8369` to `0.5994` when used without the physics term. The full model achieves the best reported MSE, `0.5801`, suggesting a complementary benefit from combining reversible normalization with thermodynamic regularization.

The physics-only variant produces an MSE of `3.9302`, which is higher than the baseline under this protocol. This result should be interpreted objectively: physical regularization alone does not guarantee improved predictive accuracy and may require appropriate normalization, loss weighting, or calibration to the source and target regimes. The full-model result motivates further investigation of the interaction between normalization and physical constraints.

## Repository Layout

```text
.
├── README.md
├── models/
│   ├── patch_transformer.py       # Patch-based Transformer backbone
│   ├── revin.py                   # Reversible Instance Normalization
│   └── physics_loss.py            # Thermodynamic regularization objective
├── utils/
│   ├── data.py                    # Dataset loading and preprocessing
│   ├── metrics.py                 # Forecasting metrics
│   ├── reproducibility.py         # Seed and deterministic-run utilities
│   └── visualization.py            # Forecast and ablation visualizations
├── scripts/
│   ├── run_ablation.py             # Deterministic ablation experiment
│   ├── train.py                    # Model training entry point
│   └── evaluate.py                 # Cross-station evaluation
├── assets/
│   ├── figures/                    # Research figures
│   └── tables/                     # Exported result tables
├── requirements.txt
└── configs/                        # Optional experiment configurations
```

The exact contents may evolve as the research implementation develops. The layout separates model components, reusable utilities, executable experiment scripts, and publication assets.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/srijitbanerjee2026-star/physics-patch-transformer.git
cd physics-patch-transformer
```

### 2. Create and activate a virtual environment

Using Python's built-in virtual-environment tooling:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a CUDA-enabled PyTorch installation is required, install the appropriate PyTorch build for the target CUDA version before installing the remaining project dependencies.

### 4. Run the deterministic ablation study

```bash
python scripts/run_ablation.py \
  --source ETTm1 \
  --target ETTm2 \
  --epochs 30 \
  --seed 42
```

The ablation script should evaluate the following configurations:

```text
1. Baseline:    no RevIN, no physics-informed loss
2. Variant A:  RevIN only
3. Variant B:  physics-informed loss only
4. Full Model:  RevIN + physics-informed loss
```

Use the command-line help to inspect implementation-specific options:

```bash
python scripts/run_ablation.py --help
```

## Reproducibility Protocol

To reproduce the reported ablation matrix as closely as possible:

- Use random seed `42`.
- Train each configuration for `30` epochs.
- Keep the data split and preprocessing pipeline fixed across variants.
- Use the same forecast horizon, input length, optimizer, learning-rate schedule, and batch size for every configuration.
- Disable target-domain fine-tuning in the ETTm1 → ETTm2 zero-shot evaluation.
- Record the software environment and hardware configuration.
- Use deterministic framework settings where supported by the selected hardware backend.

Exact numerical reproduction can depend on the Python version, deep-learning framework version, BLAS implementation, GPU architecture, and data preprocessing details.

## Evaluation Setting

The principal experiment is a cross-station transfer task:

```text
Train:      ETTm1
Evaluate:   ETTm2
Adaptation: Zero-shot; no target-domain fine-tuning
Objective:  Long-horizon time-series forecasting
Metric:     Mean squared error (MSE)
```

Additional metrics such as MAE, RMSE, and horizon-wise error profiles may be added for a more complete assessment of accuracy and stability.

## Scientific Scope and Limitations

This repository is intended for research and experimental use. The reported ablation results establish empirical behavior for the stated deterministic protocol; they should not be interpreted as universal evidence that either RevIN or physics-informed learning will improve every forecasting task.

Important limitations include:

- Results are reported for a specific ETTm1 → ETTm2 transfer direction.
- The physics-informed loss depends on the selected thermodynamic formulation and its weighting.
- A single seed and a 30-epoch budget do not characterize variance across all training conditions.
- Additional stations, horizons, seeds, and statistical significance tests are needed for broader claims.
- Physical plausibility should be assessed using domain-specific diagnostics in addition to aggregate prediction error.

## Citation

If you use this repository or build upon the method, please cite the associated manuscript. Replace the placeholder bibliographic fields with the final publication metadata when available:

```bibtex
@thesis{banerjee2026physics,
  author      = {Banerjee, Srijit},
  title       = {Physics-Informed Patch Transformer with RevIN for Zero-Shot Cross-Station Time-Series Forecasting},
  school      = {<Institution Name>},
  type        = {M.Tech. thesis},
  year        = {2026},
  address     = {<City, Country>},
  url         = {https://github.com/srijitbanerjee2026-star/physics-patch-transformer}
}
```

## License

Add the project license and any dataset-specific usage terms before distributing research code or derived artifacts.

## Acknowledgements

This work is developed as an M.Tech research project on robust time-series forecasting, OOD generalization, and physics-informed machine learning. Dataset use and redistribution should comply with the terms of the original ETT dataset and any associated benchmark license.
