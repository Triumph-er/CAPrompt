# ICDE main results

`tables/table_iii_overall_results.csv` transcribes Table III (PDF page 8) of the supplied ICDE manuscript, *CAPrompt: Context-Aware Affine Prompt Calibration for Cross-Dataset Few-Shot Traffic Forecasting*. It preserves the printed two-decimal values. Each row is one dataset/method pair, and column suffixes identify the target-training ratio (1%, 5%, or 10%). MAPE values are in percent. These are paper-level aggregate results, not per-seed records.

Empty fields correspond to dashes in the paper: STAEformer was not evaluated on PEMS03 and PEMS07 due to GPU memory limits, and TPB has no reported MAPE. Empty fields are not zero. Lower is better for all three metrics.
