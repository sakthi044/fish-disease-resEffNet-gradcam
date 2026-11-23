# Res-EffNet with Grad-CAM: fish disease classification and affected-area estimation

Code and results for the paper *Res-EffNet with Grad-CAM: Interpretable Deep Ensemble Learning for Fish Disease Diagnosis and Affected-Area Severity Estimation*.

EfficientNet-B0, ResNet-152 and DenseNet-121 are fine-tuned on the Freshwater Fish Disease Aquaculture in South Asia dataset. The notebooks also check the Test folder for near-duplicates of Train images, turn Grad-CAM maps into an affected-area percentage and a severity tier, and test robustness to image changes.

## Dataset

The dataset is public and is not included here: Biswas, "Freshwater fish disease aquaculture in south asia", Kaggle:
https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia

Set `DATA_DIR` at the top of each notebook to the folder that contains `Train` and `Test`.

The released split is used as it is: Train (1,750 images) and Test (700 images). The Train folder is split 80:20 with seed 42 into 1,400 training and 350 validation images.

**Known issue with the data.** Every test image has a perceptually identical copy in the Train folder (pHash distance 0). 586 of the 700 test images have a copy among the 1,400 training images; 114 do not. Report accuracy on the 114 as well as on all 700 (see `02_duplicates_and_figures.ipynb` and the paper, Section IV-A).

## Contents

| Path | What it is |
|---|---|
| `01_main_pipeline.ipynb` | Setup and split; training of EfficientNet-B0 (seeds 1-5), ResNet-152 and DenseNet-121 (seeds 1-3); McNemar test of the ensemble; feature-similarity duplicate check; Grad-CAM area and severity (healthy control, cut-off sensitivity); robustness tests; calibration error. Produces `results/seed_results.csv`, `results/gradcam_areas.csv`, `results/robustness.csv`. |
| `02_duplicates_and_figures.ipynb` | Perceptual-hash duplicate check (N, M and accuracy on the images with no training copy), learning curves, Grad-CAM overlays, ROC and precision-recall curves, confusion matrix, AUC and mAP. |
| `results/` | CSV files used for Tables II and III and the robustness numbers in the paper. |
| `figures/` | Confusion matrix and duplicate-pair figure. |
| `earlier_exploratory_runs/` | Single-seed notebooks from earlier experiments (for example ResNet-152 with Adam and learning rate 1e-4, and the 99.43% ensemble run). They are **not** used in any table of the paper and may use a different recipe or split from the main pipeline. |

Run `01` first. Notebook `02` needs the checkpoints and predictions that `01` saves.

## Reproducibility note

EfficientNet-B0 results were identical across the runs we compared. ResNet-152, DenseNet-121 and the ensemble were not bit-for-bit repeatable between Kaggle sessions (GPU non-determinism), so their per-seed accuracies can differ by a few images. The numbers in the paper's Table II are those in `results/seed_results.csv`. The printed outputs saved inside `02_duplicates_and_figures.ipynb` for the ResNet-152 and DenseNet-121 runs come from an earlier session and may differ from that CSV.

## Training settings

| Setting | Value |
|---|---|
| Input size | 224 x 224, ImageNet mean and standard deviation |
| Augmentation (training only) | random horizontal and vertical flip, rotation up to 15 degrees |
| Optimiser | AdamW, learning rate 3e-5, weight decay 1e-4 |
| Batch size, epochs | 16, 30 |
| Scheduler | ReduceLROnPlateau on validation accuracy, factor 0.5, patience 3 |
| Checkpoint | best validation accuracy |
| Seeds | 1-5 for EfficientNet-B0, 1-3 for the other models |
| Grad-CAM | last convolutional block of EfficientNet-B0 (`model.features[-1]`), `pytorch-grad-cam`, CAM threshold 0.5 |
| Severity | Mild: area < 10%; Moderate: 10% to < 25%; Severe: 25% or more (set by hand, not validated by experts) |
| Hardware | one Tesla T4 GPU (Kaggle) |

## Results files

| File | Content |
|---|---|
| `results/seed_results.csv` | Best validation accuracy, best epoch and test accuracy for each model and seed |
| `results/gradcam_areas.csv` | Grad-CAM area at CAM thresholds 0.3 to 0.7, predicted class and confidence for each of the 700 test images |
| `results/robustness.csv` | Accuracy of the five EfficientNet-B0 models under each image change (mean and SD) |

Model checkpoints and the softmax predictions (`.npy`) are not included here.

## Requirements

Python 3, PyTorch and torchvision, `grad-cam` (pytorch-grad-cam), scikit-learn, statsmodels, imagehash, pandas, numpy, matplotlib, seaborn.
Add the exact versions from your environment: `pip freeze | grep -iE "torch|grad-cam|scikit|statsmodels|imagehash"`.

## Limits

- The affected-area percentage measures where the classifier looks. It has not been validated against expert lesion annotations, and the severity cut-offs have no clinical justification.
- The advisory text in the paper (Table I) is general guidance, not a treatment prescription. Treatment decisions belong to a veterinarian.

## Citation and licence

Citation: <add after acceptance>. Licence: <add, for example MIT for the code>. The dataset has its own licence on Kaggle.
