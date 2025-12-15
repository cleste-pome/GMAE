# MvAE
Source code of "MvAE: Learning Disentangled Representations with Cross-view Adversarial Alignment for Multi-View Clustering"

## 1. Dataset
It can be got from: https://github.com/wangsiwei2010/awesome-multi-view-clustering

## 2. Run
Run the main.py on handwritten dataset with the followding command:

```shell
python main.py
```

### 2.1 Dataset Preprocessing

The dataset preprocessing implements the following functions: misaligned views, random views with missing values, and random views with noise.

```py
missing_ratio = 0.5
dataset.addMissing(index, missing_ratio)  # Select samples based on the missing ratio, then randomly select (1 to view-1) views to perform the missing data process (set all data to zero).
ratio_conflict = 0.4
dataset.addConflict(index, conflict_ratio)  # Select samples based on the conflict ratio, then randomly replace the data of one view with the same view data from a sample of another category.
ratio_noise = 0.1
sigma = 0.5  # 'sigma': the standard deviation of the noise
dataset.addNoise(index, noise_ratio, sigma)  # Select samples based on the noise ratio, then randomly select (1 to view-1) views to add Gaussian noise.
```

### 2.2 Evaluation Metrics

The evaluation metrics derived from the test outputs for each dataset are meticulously stored in respective files within the logs directory. Concurrently, comprehensive dataset metadata, including pertinent details, is systematically logged and preserved in logs/datasetInfo.csv, ensuring a thorough archival and easy retrieval process.
