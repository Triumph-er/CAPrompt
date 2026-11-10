# PeMS dataset documentation

The experiments use four five-minute traffic-flow benchmarks: PEMS03, PEMS04, PEMS07, and PEMS08. They are derived from the [Caltrans Performance Measurement System (PeMS)](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source), described by [Chen et al. (2001)](https://doi.org/10.3141/1748-12). See the [STSGCN project](https://github.com/Davidham3/STSGCN) for upstream benchmark information. These links are upstream references, not downloads of the exact processed splits used in this study.

## Files provided

- [`pems_split_summary.csv`](pems_split_summary.csv) records the dataset sizes, the train/validation/test window counts, and the target adaptation counts for the 1%, 5%, and 10% settings.
- [`processed_npz_manifest.csv`](processed_npz_manifest.csv) records the relative paths, byte sizes, SHA-256 checksums, and array metadata of the exact processed split files used in the experiments. The manifest does not contain traffic measurements.

## Expected local format

The local experimental data use the layout `data/<dataset>/{train,val,test}.npz`. Each archive stores `x` (windows × 12 history steps × sensors × 3 channels), `y` (windows × 12 prediction steps × sensors × 1 channel), `x_offsets` (−11 to 0), and `y_offsets` (1 to 12). Target adaptation uses the first chronologically ordered windows from the target training split, with counts specified in `pems_split_summary.csv`. The full target validation split is used for checkpoint selection, and the test split is reserved for final evaluation.

