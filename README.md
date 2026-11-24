# Employee Attrition Analysis

**Course:** CSET211 Statistical Machine Learning, Bennett University
**Author:** Vanshi Aneja (S25CSEU0828)
**Faculty:** Dr. Basab Nath

A machine learning project that studies which factors are linked to employee attrition, compares six
classification models, finds employee segments with K-Means, and includes a Streamlit dashboard and a
data-cleaning module that is tested on deliberately messy data.

## Project files

| File | What it contains |
|---|---|
| `EmployeeAttrition.ipynb` | Full analysis: preprocessing, EDA, six models, cross-validation, K-Means segmentation |
| `app.py` | Streamlit dashboard (7 pages) |
| `cleaning.py` | Data quality report, cleaning function and a function that corrupts data for testing |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | IBM HR Analytics dataset (Kaggle), 1470 employees |
| `requirements.txt` | Python libraries needed |
| `.streamlit/config.toml` | Dashboard theme |

## Dataset

IBM HR Analytics Employee Attrition & Performance (Kaggle): 1470 employees, 35 columns, 16.1% attrition.
The dataset is synthetic and imbalanced, so models are judged on recall and F1-score for the "Left" class,
not accuracy.

## Methods

1. **Preprocessing:** check missing values and duplicates, drop constant and ID columns, one-hot encoding,
   stratified 80/20 split, standardization (fit on training data only).
2. **Models:** Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, SVM.
3. **Evaluation:** test-set confusion matrices and 5-fold stratified cross-validation.
4. **Segmentation:** K-Means (K = 5) on 11 selected features, attrition rate measured per segment.
5. **Messy-data handling:** `cleaning.py` reports and fixes missing values, duplicates, placeholder text and
   negative values, and logs every action.

## Key results (5-fold cross-validation, "Left" class)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| SVM (RBF kernel) | 0.484 | 0.570 | 0.523 |
| Logistic Regression | 0.366 | 0.713 | 0.483 |
| Naive Bayes | 0.360 | 0.637 | 0.458 |
| Decision Tree | 0.372 | 0.574 | 0.449 |
| Random Forest | 0.878 | 0.118 | 0.204 |
| KNN (K = 7) | 0.608 | 0.101 | 0.173 |

K-Means found a high-risk segment of overtime-heavy junior employees with 34.2% attrition, holding nearly
half of all leavers. Results show association, not cause.

## Run the dashboard

```
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, `cleaning.py`, the CSV and the `.streamlit` folder in the same folder.

## Run the notebook

Open `EmployeeAttrition.ipynb` in Google Colab, upload the CSV, and run all cells from top to bottom.
