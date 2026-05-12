# 🩺 Cirrhosis Outcome Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

This project applies machine learning techniques to predict liver cirrhosis patient outcomes using clinical observations, physical symptoms, and laboratory test results.

The project builds a complete end-to-end machine learning workflow starting from raw medical data and ending with a tuned classification model capable of identifying different patient outcomes.

The workflow includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Missing value handling
* Feature encoding and scaling
* Classification modeling using Random Forest
* Hyperparameter tuning using GridSearchCV
* Handling class imbalance using SMOTE
* Permutation feature importance analysis
* Explanatory stakeholder-focused visualizations
* Feature engineering using clustering techniques
* Feature selection using regularized machine learning methods

The final objective is to support healthcare analytics by identifying the most important indicators associated with liver disease severity and patient survival outcomes.

---

## 🎯 Problem Statement

Given patient clinical information and laboratory measurements, predict the patient's outcome category:

* `C` → Censored / Alive
* `CL` → Liver Transplant
* `D` → Death

This is a **multi-class classification problem** with an imbalanced target variable.

---

## 📂 Dataset Information

### Dataset Summary

* Dataset: **Cirrhosis Prediction Dataset**
* Total Records: **418**
* Features: **19 input features + 1 target**
* Target Variable: `Status`

Each row represents one cirrhosis patient and contains information about:

* Liver function
* Blood test measurements
* Physical symptoms
* Disease severity
* Patient condition and survival

---

## 🧬 Understanding the Dataset Features

### 👤 Patient Information

| Feature  | Meaning                          | Medical Importance                                                         |
| -------- | -------------------------------- | -------------------------------------------------------------------------- |
| `Age`    | Patient age (in days)            | Older patients may have higher complications and worse disease progression |
| `Sex`    | Male or Female                   | Liver disease progression may differ biologically between genders          |
| `Drug`   | Treatment type received          | Treatment may affect survival and disease progression                      |
| `N_Days` | Survival days after registration | Lower survival duration often indicates severe disease                     |

---

### 🩺 Clinical Symptoms

| Feature        | Meaning                             | Why It Matters                                          |
| -------------- | ----------------------------------- | ------------------------------------------------------- |
| `Ascites`      | Fluid accumulation in abdomen       | Strong indicator of advanced liver disease              |
| `Hepatomegaly` | Enlarged liver                      | Associated with liver inflammation and cirrhosis        |
| `Spiders`      | Spider-shaped blood vessels on skin | Common symptom of chronic liver dysfunction             |
| `Edema`        | Swelling severity                   | Indicates fluid retention and worsening liver condition |

---

### 🧪 Laboratory Measurements

| Feature         | Meaning                                   | Medical Interpretation                                  |
| --------------- | ----------------------------------------- | ------------------------------------------------------- |
| `Bilirubin`     | Waste product processed by liver          | High values indicate poor liver function and jaundice   |
| `Albumin`       | Protein produced by liver                 | Low levels suggest liver damage                         |
| `Copper`        | Copper concentration in blood             | Elevated levels are linked to liver dysfunction         |
| `Alk_Phos`      | Liver enzyme level                        | High values may indicate bile duct or liver damage      |
| `SGOT`          | Liver enzyme associated with liver injury | Elevated values indicate liver inflammation             |
| `Tryglicerides` | Blood fat measurement                     | Abnormal values may reflect metabolic abnormalities     |
| `Platelets`     | Blood platelet count                      | Low values are associated with advanced cirrhosis       |
| `Prothrombin`   | Blood clotting time                       | Higher values indicate impaired liver clotting function |
| `Cholesterol`   | Cholesterol level                         | Liver dysfunction can affect cholesterol metabolism     |

---

### 📈 Disease Severity

| Feature | Meaning                                                         |
| ------- | --------------------------------------------------------------- |
| `Stage` | Cirrhosis stage from 1 → 4 (higher stage = more severe disease) |

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Jupyter Notebook

---

## 🧹 Data Cleaning & Preprocessing

### 🧩 Missing Values

Several medical test features contained missing values, especially:

* `Cholesterol`
* `Tryglicerides`
* `Copper`
* `SGOT`
* `Alk_Phos`

Handling strategy:

* Numerical Features → Median Imputation
* Categorical Features → Most Frequent Imputation

Median imputation was selected because medical data contains strong skewness and multiple outliers.

---

### 🗂️ Feature Types

#### Numerical Features

* `Age`
* `Bilirubin`
* `Albumin`
* `Copper`
* `Platelets`
* `Prothrombin`
* `SGOT`
* `Cholesterol`
* `Tryglicerides`

#### Nominal Features

* `Drug`
* `Sex`
* `Ascites`
* `Hepatomegaly`
* `Spiders`

#### Ordinal Feature

* `Edema`

Encoded as:

```text
N < S < Y
```

---

### ⚙️ Encoding & Scaling

* One-Hot Encoding → Nominal Features
* Ordinal Encoding → `Edema`
* StandardScaler → Numerical Features

---

### 🔄 Preprocessing Pipeline

A reusable preprocessing pipeline was developed using `ColumnTransformer` to ensure consistent transformations across training and testing data.

The pipeline included:

* `SimpleImputer` for missing value handling
* `StandardScaler` for numerical scaling
* `OrdinalEncoder` for ordered categorical features
* `OneHotEncoder` for nominal categorical variables

This structure improved modularity, reproducibility, and integration with the machine learning workflow.

---

## 🔍 Exploratory Data Analysis (EDA)

### 📌 Key Findings

* Dataset contains significant class imbalance, especially for transplant cases (`CL`)
* Most biochemical markers are heavily right-skewed
* Advanced cirrhosis stages dominate the dataset
* Severe outcomes are associated with abnormal liver-function biomarkers

---

### 📊 Important Medical Insights

* Higher `Bilirubin` levels strongly correlate with severe patient outcomes
* Lower `Albumin` levels indicate declining liver functionality
* Increased `Prothrombin` time reflects impaired blood clotting caused by liver failure
* Low `Platelets` are associated with advanced cirrhosis complications and portal hypertension

---

# 🛠️ Feature Engineering

Feature engineering was performed to enhance the predictive capability of the model and capture hidden structure within the medical dataset.

The project incorporated:

* Clustering-based feature generation using KMeans
* Dimensionality reduction using feature selection
* Class balancing using SMOTE oversampling

These steps were integrated into the machine learning pipeline to improve generalization and model interpretability.

---

## 🔗 Clustering Using KMeans

### 📌 Objective

KMeans clustering was applied as an unsupervised feature engineering technique to identify hidden patient subgroups within the dataset.

The generated cluster labels were then added as a new predictive feature.

---

### ⚙️ Clustering Process

Steps performed:

1. Applied preprocessing pipeline on the dataset
2. Tested multiple cluster values (`k`)
3. Evaluated clustering quality using:

   * Silhouette Score
   * Inertia (Elbow Method)
4. Selected:

   ```text
   k = 2
   ```
5. Added cluster assignments as a new feature named `Cluster`

---

### 📊 Clustering Insight

The engineered `Cluster` feature later appeared among the most important features during permutation importance analysis.

This suggests that clustering successfully captured meaningful hidden structure related to disease severity and patient outcomes.

---

## 🎯 Feature Selection

### 📌 Objective

Feature selection was performed to reduce dimensionality, remove less informative variables, and improve model interpretability.

---

### ⚙️ Method Used

The project used:

```python
SelectFromModel(LogisticRegression(penalty='l1'))
```

L1 regularization automatically shrinks less important feature coefficients toward zero, allowing the model to retain only the most relevant predictors.

---

### 📊 Feature Selection Impact

Benefits observed:

* Reduced feature space complexity
* Improved interpretability
* Lower redundancy between encoded variables
* More efficient model training

However, a slight decrease in overall predictive performance was observed after aggressive feature reduction.

---

## ⚖️ Handling Class Imbalance with SMOTE

### 📌 Problem

The dataset contained severe imbalance between outcome classes, particularly the `CL` (Liver Transplant) category.

This caused the model to become biased toward majority classes.

---

### 🧪 Solution

SMOTE (Synthetic Minority Oversampling Technique) was applied **only on the training dataset** to prevent data leakage.

SMOTE generated synthetic samples for minority classes and balanced the class distribution before model training.

---

### 📊 Effect of SMOTE

SMOTE improved:

* Minority class learning
* Recall for transplant cases (`CL`)
* Overall prediction balance across classes

Although overall accuracy did not increase significantly, the model became more clinically balanced and fair.

---

## 🤖 Machine Learning Workflow

The project followed this workflow:

1. Data Cleaning
2. Exploratory Data Analysis
3. Missing Value Handling
4. Feature Encoding & Scaling
5. Feature Engineering with KMeans
6. Feature Selection
7. Train-Test Split
8. SMOTE Oversampling for Class Imbalance
9. Random Forest Modeling
10. Hyperparameter Tuning using GridSearchCV
11. Model Evaluation
12. Permutation Importance Analysis
13. Explanatory Visualizations

---

## 🌲 Model Used

### Random Forest Classifier

Random Forest was selected because it:

* Handles mixed feature types effectively
* Performs strongly on tabular healthcare datasets
* Captures nonlinear feature interactions
* Is robust against noise and outliers

---

## 📊 Model Comparison

| Model Version             | Accuracy | Macro F1 | Key Characteristics                                                         |
| ------------------------- | -------- | -------- | --------------------------------------------------------------------------- |
| Baseline Random Forest    | ~0.76    | ~0.71    | Strong overall accuracy but weaker minority class learning                  |
| Engineered Pipeline Model | ~0.75    | ~0.68    | Better balance, improved interpretability, stronger minority class handling |

---

### 📌 Interpretation

Although the engineered pipeline produced slightly lower overall metrics, it offered several important advantages:

* Better handling of minority transplant cases (`CL`)
* Improved class balance
* More interpretable feature space
* Reduced dimensionality
* Better clinical insight extraction

This highlights an important machine learning principle in healthcare analytics:

> Higher accuracy alone does not always imply a better clinical model.

---

## 📈 Model Performance

### 🚨 Initial Model

The default Random Forest model achieved:

* Training Accuracy = 100%
* Poor test generalization

➡️ Severe overfitting detected.

---

### ⚙️ After Hyperparameter Tuning

Best parameters:

```python
{
 'classifier__class_weight': None,
 'classifier__max_depth': 5,
 'classifier__max_features': 'sqrt',
 'classifier__min_samples_leaf': 2,
 'classifier__min_samples_split': 5,
 'classifier__n_estimators': 100
}
```

Overfitting was reduced, but minority class prediction remained weak.

---

### 🚀 Final Engineered Model

Best optimized configuration:

```python
{
 'n_estimators': 200,
 'max_depth': 5,
 'max_features': 'sqrt',
 'min_samples_split': 2,
 'min_samples_leaf': 1
}
```

The final pipeline integrated:

* Full preprocessing
* KMeans clustering
* Feature selection
* SMOTE oversampling
* Hyperparameter-tuned Random Forest

---

### ⚖️ After Applying SMOTE

SMOTE significantly improved minority class learning and produced more balanced predictions.

### 📊 Final Test Performance

| Metric            | Baseline Model | Final Engineered Model           |
| ----------------- | -------------- | -------------------------------- |
| Accuracy          | 0.76           | 0.75                             |
| Macro F1-Score    | 0.71           | 0.68                             |
| Weighted F1-Score | 0.76           | More balanced class distribution |

---

### 📌 Key Improvements

* Better prediction of transplant cases (`CL`)
* Improved balance across all target classes
* Reduced model bias toward majority outcomes
* More stable train-test generalization
* Increased interpretability through feature engineering

---

## 📊 Permutation Feature Importance

The most important predictive features were:

1. Bilirubin
2. Age
3. N_Days
4. Prothrombin
5. Platelets
6. Copper
7. SGOT
8. Edema
9. Ascites
10. Cluster

These features are medically meaningful and strongly associated with liver disease severity and survival outcomes.

The inclusion of the engineered `Cluster` feature among top predictors demonstrates that unsupervised clustering successfully captured useful hidden patterns within the patient population.

### 📷 Feature Importance Visualization

<img width="1214" height="683" alt="image" src="https://github.com/user-attachments/assets/46db2f19-76d4-4c73-9a60-96d531ab8a2c" />

---

## 📌 Explanatory Visualization 1

### Average Bilirubin by Patient Outcome

<img width="1046" height="688" alt="image" src="https://github.com/user-attachments/assets/e829430d-f27a-4d86-914e-6afb4c42b839" />

### Insight

Patients in the `D` (death) category show the highest average bilirubin levels.

Since bilirubin reflects the liver’s ability to process waste products, elevated values indicate severe liver dysfunction and worsening prognosis.

This confirms bilirubin as one of the strongest indicators of cirrhosis severity.

---

## 📌 Explanatory Visualization 2

### Average Prothrombin Time by Patient Outcome

<img width="1058" height="691" alt="image" src="https://github.com/user-attachments/assets/48e833bb-5e22-4f00-80d7-3758f9565a60" />

### Insight

Patients with severe outcomes tend to have higher prothrombin times.

Prothrombin measures blood clotting speed, and impaired clotting is a common complication of advanced liver disease.

This visualization demonstrates how worsening liver function affects the body's clotting ability.

---

## 🧠 Final Insights

* Liver-function biomarkers strongly influence cirrhosis outcome prediction
* Class imbalance significantly affects healthcare machine learning models
* SMOTE successfully improved minority class detection
* Bilirubin and Prothrombin emerged as the strongest clinical indicators
* Clustering introduced meaningful hidden patient-group information
* Feature selection improved interpretability and reduced dimensionality
* The model learned medically meaningful patterns rather than random relationships

---

# 📌 Final Conclusion

A complete end-to-end machine learning pipeline was developed to predict cirrhosis patient outcomes using clinical and laboratory data.

The project successfully integrated:

* Advanced preprocessing pipelines
* Feature engineering using KMeans clustering
* Feature selection with L1-regularized models
* SMOTE oversampling for class balancing
* Hyperparameter optimization using GridSearchCV
* Permutation importance analysis
* Clinically interpretable visualizations

Although feature engineering and balancing techniques slightly reduced overall accuracy, they improved model fairness, interpretability, and minority-class learning — which are critical objectives in healthcare machine learning systems.

The final model demonstrated the ability to identify medically meaningful patterns associated with disease progression, transplant necessity, and mortality risk.

This project highlights how machine learning can support healthcare analytics by transforming raw clinical data into actionable predictive insights.

---

## 👩‍💻 Author

Maryam Skaik
Computer Science Graduate | Teaching Assistant | Data Science Enthusiast
