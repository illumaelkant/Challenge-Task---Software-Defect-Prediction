# Software Defect Prediction with CatBoost - Challenge Task

This project provides a machine learning model for predicting software defects using the CatBoost algorithm. It is designed to work well with imbalanced datasets, especially those from the PROMISE repository.

## Package Requirements

- Python 3.12
- Anaconda 24.9.2
- CatBoost

### Install dependencies

```bash
pip install catboost
```

### Dataset

We use datasets from the PROMISE repository, NASA Dataset and other. These datasets contain software modules labeled as defective (>1) or non-defective (0).
We transform the problem into binary classification

Metrics printed:

    Accuracy

    F1-Score

    ROC AUC

    G-Mean

    Confusion Matrix

 Why CatBoost?

CatBoost handles:

    Categorical features natively

    Imbalanced data effectively (with class weights)

    High-dimensional structured data well

It's particularly useful for software engineering data where features may have hierarchical or categorical patterns or the training data is unbalanced.
Example Results are shown in ```/Finel_Result```

