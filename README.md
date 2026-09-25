# Customer Category Affinity Prediction

A machine-learning classification project for identifying customers who are likely to show **affinity toward a product category** based on customer behavior, engagement, purchase history, value, and category-preference signals.

## Business Problem

In an e-commerce or campaign-management environment, customers have different purchasing patterns and category preferences.

The business objective is to answer:

> **Is this customer likely to have positive affinity toward a product category?**

The current dataset contains a `Preferred Category` feature, together with behavioral and engagement features. The trained model predicts the binary `Target` label associated with the affinity dataset.

### Example business flow

```text
Customer behavioral data
        +
Category-preference signals
        ↓
Feature preprocessing
        ↓
Gradient Boosting Classifier
        ↓
Affinity probability
        ↓
Target: 0 / 1
```

## Important Scope Note

The current notebook is **not an arbitrary `customer + any requested category -> affinity` API**. The training data contains `Preferred Category`, which is used as one of the categorical input features.

Therefore, this repository accurately represents the executed notebook as a **customer affinity classification model using category-preference information**.

A production system that must score the same customer against *every possible category* should explicitly construct a customer-category training dataset, include the candidate category as an input, and generate one score per customer-category pair.

## Dataset

The supplied dataset contains:

- **50,000 customer records**
- **21 columns before one-hot encoding**
- Binary target: `Target`
- Positive class: 866 records
- Negative class: 49,134 records
- Positive-class rate: approximately **1.73%**

### Main features

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `Gender` | Customer gender |
| `Region / State` | Geographic region |
| `Purchase Count` | Number of purchases |
| `Total Spend` | Customer spending |
| `Average Order Value` | Average order value |
| `Average Items Per Order` | Average items purchased per order |
| `Last Purchase Days` | Days since latest purchase |
| `Tenure Days` | Customer tenure |
| `Preferred Category` | Customer's preferred product category |
| `Category Diversity` | Number/diversity of categories purchased |
| `Preferred Channel` | Preferred customer channel |
| `Session Count` | Number of sessions |
| `Engagement Score` | Customer engagement signal |
| `Campaign Engagement Score` | Campaign engagement signal |
| `Email Click Rate` | Email click behavior |
| `Campaign Conversion Rate` | Campaign conversion behavior |
| `Loyalty Score` | Loyalty signal |
| `Value Score` | Customer value signal |
| `Purchase Frequency` | Purchase frequency |
| `Target` | Binary affinity target |

## Class Imbalance

The target is highly imbalanced:

```text
Class 0: 49,134  (~98.27%)
Class 1:    866  (~1.73%)
```

Because of this imbalance, accuracy alone should not be used to judge the model. Precision, recall, F1-score and ROC-AUC are also reported.

## Machine Learning Workflow

The notebook follows this workflow:

1. Load the affinity dataset
2. Inspect data types and target distribution
3. Handle categorical features
4. One-hot encode categorical columns
5. Define feature matrix and target
6. Split data using a stratified train/test split
7. Train multiple classification models
8. Compare model performance
9. Tune Gradient Boosting using Optuna
10. Evaluate the optimized model
11. Save the trained model with Joblib
12. Reload the model and validate prediction consistency

### Preprocessing

Categorical columns are one-hot encoded using:

```python
pd.get_dummies(..., drop_first=True, dtype=int)
```

The resulting feature matrix contains **42 model features**.

## Models Evaluated

The notebook evaluates:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Bayesian-optimized Gradient Boosting

## Baseline Results

Results recorded by the executed notebook:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 99.80% | 95.81% | 92.49% | 94.12% | 99.98% |
| Random Forest | 99.46% | 94.07% | 73.41% | 82.47% | 99.88% |
| Gradient Boosting | 99.62% | 94.70% | 82.66% | 88.27% | 99.64% |
| XGBoost | 99.65% | 94.81% | 84.39% | 89.30% | 99.94% |
| LightGBM | 99.61% | 92.95% | 83.82% | 88.15% | 99.92% |
| CatBoost | 99.67% | 96.05% | 84.39% | 89.85% | 99.93% |

## Bayesian Optimization

Gradient Boosting was tuned with Optuna.

The optimization objective was ROC-AUC.

Best recorded parameters:

```text
n_estimators      = 459
learning_rate     ≈ 0.048872
max_depth         = 4
min_samples_split = 2
min_samples_leaf  = 3
subsample         ≈ 0.803652
random_state      = 42
```

The notebook used 7 optimization trials.

### Optimized Model Result

Recorded test-set result from the notebook:

| Metric | Result |
|---|---:|
| Accuracy | 99.74% |
| Precision | 96.23% |
| Recall | 88.44% |
| F1 Score | 92.17% |
| ROC-AUC | ~99.92% |

Confusion matrix:

```text
TN = 9821
FP =    6
FN =   20
TP =  153
```

The saved model was reloaded successfully and produced:

```text
Loaded Model Accuracy: 0.9974
```

## Important Evaluation Caveat

The current notebook has a methodological issue that should be fixed before presenting the reported ROC-AUC as a final unbiased production/test estimate.

The Optuna objective directly evaluates each trial on `X_test` / `y_test` and uses that test ROC-AUC to select the best hyperparameters. This means the test set participates in model selection.

In addition, categorical preprocessing is performed before the train/test split.

For a production-grade version, the recommended approach is:

```text
Raw data
   ↓
Train / validation / test split
   ↓
Fit preprocessing only on training data
   ↓
Cross-validation on training data
   ↓
Hyperparameter tuning
   ↓
Final model
   ↓
One-time evaluation on untouched test set
```

The metrics in this README are therefore reported as **the notebook's recorded results**, not as an independently validated production benchmark.

## Model Artifact

The trained model is included at:

```text
models/Affinity_model_Bayesian_Optimized.pkl
```

It is a scikit-learn `GradientBoostingClassifier` trained on the notebook's 42-feature encoded matrix.

## Repository Structure

```text
category-affinity-prediction/
├── data/
│   └── Affinity_data_no_leakage_target.csv
├── models/
│   └── Affinity_model_Bayesian_Optimized.pkl
├── notebooks/
│   └── category_affinity.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Loka746/Category_Affinity.git
cd Category_Affinity

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter notebook notebooks/category_affinity.ipynb
```

Run the notebook from top to bottom to reproduce the preprocessing, model training, evaluation, optimization and model-saving workflow.

## Recommended Production Architecture

For a true category-affinity recommendation service, the next iteration should use a customer-category pair as the prediction unit:

```text
Customer
   +
Candidate Category
   ↓
Customer-category feature engineering
   ↓
Preprocessing Pipeline
   ↓
Affinity Model
   ↓
P(affinity)
   ↓
Rank categories for the customer
```

This enables outputs such as:

```text
Customer A
├── Electronics → 0.91
├── Fashion     → 0.34
├── Sports      → 0.12
└── Home        → 0.67
```

The probabilities above are only an illustration of the desired production interface and are **not model outputs from this repository**.

## Future Improvements

- Build explicit customer-category pair training data
- Add candidate category as a direct model input
- Use a leakage-safe preprocessing pipeline
- Use cross-validation during Optuna tuning
- Keep a completely untouched test set
- Optimize threshold based on business costs
- Add probability calibration
- Add model explainability with SHAP
- Expose inference through FastAPI
- Add batch scoring for campaign targeting
- Add monitoring for data drift and prediction drift

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- Joblib
- Jupyter Notebook

## Author

**Lokanadham K.**


# Category-Affinity
# Category-Affinity
# Category-Affinity
