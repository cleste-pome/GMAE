# GMAE
Source code of "Learning Disentangled Representations for Generalized Multi-view Clustering"

GMAE-Github
├── 1.logs_classification # 分类任务的日志与图像文件
├── 2.imgs_classification 
├── 1.logs_clustering # 聚类任务的日志与图像文件
├── 2.imgs_clustering 
├── dataset # 数据集存放目录
├── utils # 辅助功能的工具包
│ ├── dataloader.py # 数据集加载与预处理
│ ├── Logger.py # log文档打印
│ ├── metric.py # 评价指标计算
│ ├── plot.py # 绘制评价指标表格和图像
├── classification.py # 多视图分类主程序
├── clustering.py # 多视图聚类主程序
├── loss.py # 损失函数（部分）定义
├── models.py # 模型定义
└── external # 外部库

## 1. Dataset
It can be got from: https://github.com/wangsiwei2010/awesome-multi-view-clustering

## 2. Run
(1) To run the **multi-view clustering** task, use the following command:

```shell
python clustering.py
```

(2) To run the **multi-view classification** task, use the following command:

```shell
python classification.py
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
