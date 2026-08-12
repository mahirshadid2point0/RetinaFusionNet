# RetinaFusionNet

Official implementation and experimental notebooks for **RetinaFusionNet**, a multimodal framework for CIMT classification from bilateral retinal fundus images and demographic information.

The repository contains the training and evaluation pipelines used for the reported experiments, including gated demographic fusion, ungated ablations, fundus-only and single-eye experiments, seed experiments, test-time augmentation, threshold tuning, and the three-backbone probability-averaging ensemble.

## 1. Environment

The experiments were implemented with the following software environment:

| Component | Version |
|---|---|
| Python | 3.13.0 |
| PyTorch | 2.12.0+cpu |
| TorchVision | 0.27.0+cpu |
| OpenCV | 4.13.0 |
| Albumentations | 2.0.8 |
| Pillow | 12.2.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| Pandas | 3.0.3 |

GPU execution is supported by the notebooks when CUDA is available. The reported experiments used an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB VRAM and 16 GB RAM.

Install the required packages with:

```bash
pip install torch torchvision opencv-python albumentations pillow numpy scipy scikit-learn pandas matplotlib tqdm jupyter
```

For exact reproducibility, the versions listed above should be used.

## 2. Dataset

The experiments use the **China-Fundus-CIMT dataset** with the official patient-level split:

- Training: 2,603 patients
- Validation: 200 patients
- Test: 100 patients

Each patient record contains bilateral fundus images, demographic information, and the CIMT classification label.

The repository does not redistribute the dataset. The dataset and metadata must be obtained separately and placed on the local machine.

## 3. Dataset Configuration

The notebooks currently contain local path variables that must be changed before execution.

In the configuration section, update:

```python
DATA_ROOT = Path(r"<PATH_TO_CHINA_FUNDUS_CIMT_DATASET>")
META_PATH = Path(r"<PATH_TO_DATA_INFO_JSON>")
```

`DATA_ROOT` should point to the directory containing the fundus images, while `META_PATH` should point to the dataset metadata JSON file.

The official train, validation, and test split is loaded from the metadata and should not be re-split.

## 4. Repository Contents

```text
RetinaFusionNet/
├── CIMT_Improved_Pipeline-Final-Run.ipynb
├── CIMT_Improved_Pipeline-No-Gate.ipynb
├── CIMT_Improved_Pipeline-FUNDUS-Only.ipynb
├── CIMT_Improved_Pipeline-Left_Eye_Only.ipynb
├── CIMT_Improved_Pipeline-Right_Eye_Only.ipynb
├── CIMT_Improved_Pipeline-Seed64.ipynb
├── CIMT_Improved_Pipeline-Seed92.ipynb
├── checkpoints_pth/
└── .gitattributes
```

The `checkpoints_pth` directory contains the trained `.pth` model checkpoints tracked using Git LFS.

## 5. Running the Main RetinaFusionNet Experiment

Open:

```text
CIMT_Improved_Pipeline-Final-Run.ipynb
```

Run the notebook from the beginning after updating `DATA_ROOT` and `META_PATH`.

The main pipeline:

1. Loads the official patient-level train/validation/test split.
2. Crops the retinal region automatically.
3. Applies synchronized bilateral augmentation during training.
4. Extracts features using a shared-weight CNN backbone.
5. L2-normalizes and averages the bilateral eye features.
6. Encodes age and gender.
7. Applies the learned demographic gating mechanism.
8. Trains the classifier using label-smoothed, class-weighted BCE loss.
9. Uses discriminative learning rates with linear warm-up and cosine decay.
10. Selects the best checkpoint using validation ROC-AUC with early stopping.
11. Tunes the classification threshold using validation F1.
12. Applies horizontal-flip TTA to the test set.
13. Reports test performance at both `T = 0.5` and the validation-tuned threshold.
14. Computes a patient-level bootstrap 95% confidence interval for ROC-AUC.
15. Produces the three-backbone probability-averaging ensemble.

The three backbones are:

```python
BACKBONES = [
    "efficientnet_b0",
    "resnet50",
    "densenet121"
]
```

## 6. Reproducing the Reported Main Models

The main notebook trains:

- RetinaFusionNet with EfficientNet-B0
- RetinaFusionNet with ResNet50
- RetinaFusionNet with DenseNet121
- RetinaFusionNet Ensemble

The default configuration uses:

```text
Seed = 42
Image size = 224 × 224
Batch size = 32
Maximum epochs = 30
Early stopping patience = 7
Warm-up epochs = 2
Backbone learning rate = 3e-5
Fusion head learning rate = 3e-4
Weight decay = 5e-4
Dropout = 0.30
Label smoothing = 0.05
Gradient clipping = 1.0
```

The conventional classification threshold is:

```text
T = 0.5
```

The tuned threshold is selected from the validation set by maximizing F1 and then applied unchanged to the test set.

## 7. Alternative Experimental Pipelines

The repository contains separate notebooks corresponding to the reported experimental conditions.

### Ungated multimodal fusion

```text
CIMT_Improved_Pipeline-No-Gate.ipynb
```

This removes the learned demographic gate while retaining the other main pipeline components.

### Fundus-only

```text
CIMT_Improved_Pipeline-FUNDUS-Only.ipynb
```

This evaluates the image modality without demographic information.

### Left-eye only

```text
CIMT_Improved_Pipeline-Left_Eye_Only.ipynb
```

### Right-eye only

```text
CIMT_Improved_Pipeline-Right_Eye_Only.ipynb
```

### Seed 64

```text
CIMT_Improved_Pipeline-Seed64.ipynb
```

Only EfficientNet-B0 was evaluated for this seed because EfficientNet-B0 was the best-performing backbone in the main gated experiment and the additional backbone runs were restricted by computational time.

### Seed 92

```text
CIMT_Improved_Pipeline-Seed92.ipynb
```

Only EfficientNet-B0 was evaluated for the same computational-time reason.

All other reported configurations use:

```text
Seed = 42
```

## 8. Using the Provided Checkpoints

Pretrained experiment checkpoints are available under:

```text
checkpoints_pth/
```

The `.pth` files are tracked using **Git LFS**.

After cloning the repository, install Git LFS and retrieve the checkpoint files:

```bash
git lfs install
git lfs pull
```

The main RetinaFusionNet checkpoints include:

```text
best_efficientnet_b0_RetinaFusionNet_Final_Run.pth
best_resnet50_RetinaFusionNet_Final_Run.pth
best_densenet121_RetinaFusionNet_Final_Run.pth
```

Additional checkpoints correspond to the single-eye, ungated, and seed-specific experiments.

## 9. Evaluation

The main notebook evaluates each trained model using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 95% bootstrap confidence interval for ROC-AUC

The bootstrap analysis uses:

```text
2,000 patient-level resamples
Random state = 42
```

Because each test-set row corresponds to one patient, bootstrap resampling of test rows corresponds directly to patient-level resampling.

## 10. Test-Time Augmentation

Horizontal-flip TTA averages predictions from:

```text
Original image
+
Horizontally flipped image
```

The resulting probability is used for the final test evaluation.

The notebooks also contain the comparison between the TTA and no-TTA configurations.

## 11. Outputs

During execution, the notebooks create:

```text
checkpoints/
results/
```

The results directory contains per-backbone metrics, ensemble results, test predictions, and experiment summaries generated by the notebooks.

## 12. Reproducibility Notes

For reproducibility:

1. Use the official China-Fundus-CIMT patient-level split.
2. Do not re-split the dataset.
3. Set the required dataset paths before running a notebook.
4. Use the specified random seed for the corresponding experiment.
5. Keep the reported preprocessing and training configuration unchanged.
6. Select checkpoints using validation ROC-AUC only.
7. Tune thresholds using validation F1 only.
8. Evaluate the test set only after model and threshold selection are complete.

The test set contains 100 patients and is reserved for final evaluation.

## 13. Citation

If this repository or RetinaFusionNet is used in research, please cite the associated paper.
