# CAPrompt: Context-Aware Affine Prompt Calibration for Cross-Dataset Few-Shot Traffic Forecasting

This repository provides the core CAPrompt model and the main results reported in the ICDE manuscript.

- [`model.py`](model.py): PyTorch implementation of the forecasting backbone and the context-aware affine prompt calibrator.
- [`results/tables/table_iii_overall_results.csv`](results/tables/table_iii_overall_results.csv): main comparison results from Table III.
- [`data/`](data/README.md): sources, experimental split counts, and SHA-256 checksums for the prepared PEMS03, PEMS04, PEMS07, and PEMS08 files.

The processed `.npz` files are not included in this repository. Training and evaluation scripts, baseline implementations, and model checkpoints are also not included. The repository is intended for inspecting the model and reported results; it does not provide an end-to-end reproduction workflow.
