# Code for "A Field-Based Image Dataset for Automated Recognition of Okra Plant Diseases"

Dataset: https://doi.org/10.17632/tdcmkjv3yr.1 (Mendeley Data, CC BY 4.0)

## Contents
- `okra_ConvNeXt-Tiny_MobileNetV3_results_groupcv.ipynb` – integrity and duplicate checks,
  near-duplicate grouping, group-aware 5-fold training, evaluation and statistics
  (with the outputs of the run reported in the paper)
- `okra_ConvNeXt-Tiny_MobileNetV3_results_groupcv.zip` – per-fold metrics, predictions,
  classification reports, training histories and statistical tests from that run
- `augment_dataset.py` – generates Augmented_Image_Dataset (-aug-1 to -aug-5)
- `group_aware_5fold_manifest.csv` – near-duplicate groups and fold assignments
  for the 4,258 unique images used in the benchmark
- `requirements.txt` – Python package versions used in the experiments

## Reproducing the benchmark
The notebook is written for Kaggle (2 × NVIDIA T4 GPUs).
1. Download the dataset from Mendeley Data and add the `Raw_Image_Dataset` folder
   to a Kaggle notebook as an input dataset.
2. Optionally, add `group_aware_5fold_manifest.csv` as an input as well.
3. Run all cells. The notebook removes the nine exact duplicates, rebuilds the
   near-duplicate groups and the group-aware folds (seed 42) and, if the manifest
   is provided, confirms that the folds match the released assignment.

To run outside Kaggle, edit the paths in the configuration cell (Section 3).

## Regenerating the augmented images
```
pip install pillow tqdm
python augment_dataset.py --input /path/to/Raw_Image_Dataset --output augmented_dataset
```
Five variants are saved per image: horizontal flip (-aug-1), rotation by 30°
counter-clockwise (-aug-2), brightness ×0.8 (-aug-3), contrast ×1.5 (-aug-4)
and Gaussian blur, radius 0.8 px (-aug-5).

## License
Code: MIT License (see LICENSE). Dataset: CC BY 4.0.

## Citation
If you use this code or dataset, please cite:

Masum, S. M., Mondol, M. R. A., Hridoy, R. H., Habib, M. T. & Islam, M. M.
A diverse field image dataset of okra plant diseases for automated disease diagnosis
and plant pathology research. Mendeley Data V1, https://doi.org/10.17632/tdcmkjv3yr.1 (2026).
