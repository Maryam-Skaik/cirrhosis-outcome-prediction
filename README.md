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

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Missing value handling  
- Feature encoding and scaling  
- Classification modeling using Random Forest  
- Hyperparameter tuning using GridSearchCV  
- Handling class imbalance using SMOTE  
- Permutation feature importance analysis  
- Explanatory stakeholder-focused visualizations  

The final objective is to support healthcare analytics by identifying the most important indicators associated with liver disease severity and patient survival outcomes.

---

## 🎯 Problem Statement

Given patient clinical information and laboratory measurements, predict the patient's outcome category:

- `C` → Censored / Alive  
- `CL` → Liver Transplant  
- `D` → Death  

This is a **multi-class classification problem** with an imbalanced target variable.

---

## 📂 Dataset Information

### Dataset Summary

- Dataset: **Cirrhosis Prediction Dataset**
- Total Records: **418**
- Features: **19 input features + 1 target**
- Target Variable: `Status`

Each row represents one cirrhosis patient and contains information about:

- Liver function
- Blood test measurements
- Physical symptoms
- Disease severity
- Patient condition and survival

---

## 🧬 Understanding the Dataset Features

### 🔻 Patient Information

| Feature | Meaning | Medical Importance |
|------|------|------|
| `Age` | Patient age (in days) | Older patients may have higher complications and worse disease progression |
| `Sex` | Male or Female | Liver disease progression may differ biologically between genders |
| `Drug` | Treatment type received | Treatment may affect survival and disease progression |
| `N_Days` | Survival days after registration | Lower survival duration often indicates severe disease |

---

### 🔻 Clinical Symptoms

| Feature | Meaning | Why It Matters |
|------|------|------|
| `Ascites` | Fluid accumulation in abdomen | Strong indicator of advanced liver disease |
| `Hepatomegaly` | Enlarged liver | Associated with liver inflammation and cirrhosis |
| `Spiders` | Spider-shaped blood vessels on skin | Common symptom of chronic liver dysfunction |
| `Edema` | Swelling severity | Indicates fluid retention and worsening liver condition |

---

### 🔻 Laboratory Measurements

| Feature | Meaning | Medical Interpretation |
|------|------|------|
| `Bilirubin` | Waste product processed by liver | High values indicate poor liver function and jaundice |
| `Albumin` | Protein produced by liver | Low levels suggest liver damage |
| `Copper` | Copper concentration in blood | Elevated levels are linked to liver dysfunction |
| `Alk_Phos` | Liver enzyme level | High values may indicate bile duct or liver damage |
| `SGOT` | Liver enzyme associated with liver injury | Elevated values indicate liver inflammation |
| `Tryglicerides` | Blood fat measurement | Abnormal values may reflect metabolic abnormalities |
| `Platelets` | Blood platelet count | Low values are associated with advanced cirrhosis |
| `Prothrombin` | Blood clotting time | Higher values indicate impaired liver clotting function |
| `Cholesterol` | Cholesterol level | Liver dysfunction can affect cholesterol metabolism |

---

### 🔻 Disease Severity

| Feature | Meaning |
|------|------|
| `Stage` | Cirrhosis stage from 1 → 4 (higher stage = more severe disease) |

---

## ⚙️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Jupyter Notebook  

---

## 🧹 Data Cleaning & Preprocessing

### 🔻 Missing Values

Several medical test features contained missing values, especially:

- `Cholesterol`
- `Tryglicerides`
- `Copper`
- `SGOT`
- `Alk_Phos`

Handling strategy:

- Numerical Features → Median Imputation  
- Categorical Features → Most Frequent Imputation  

Median imputation was selected because medical data contains strong skewness and multiple outliers.

---

### 🔻 Feature Types

#### Numerical Features
- `Age`
- `Bilirubin`
- `Albumin`
- `Copper`
- `Platelets`
- `Prothrombin`
- `SGOT`
- `Cholesterol`
- `Tryglicerides`

#### Nominal Features
- `Drug`
- `Sex`
- `Ascites`
- `Hepatomegaly`
- `Spiders`

#### Ordinal Feature
- `Edema`

Encoded as:

```text
N < S < Y
```

---

### 🔻 Encoding & Scaling

- One-Hot Encoding → Nominal Features  
- Ordinal Encoding → `Edema`  
- StandardScaler → Numerical Features  

---

## 🔍 Exploratory Data Analysis (EDA)

### 📌 Key Findings

- Dataset contains significant class imbalance, especially for transplant cases (`CL`)
- Most biochemical markers are heavily right-skewed
- Advanced cirrhosis stages dominate the dataset
- Severe outcomes are associated with abnormal liver-function biomarkers

---

### 📊 Important Medical Insights

- Higher `Bilirubin` levels strongly correlate with severe patient outcomes  
- Lower `Albumin` levels indicate declining liver functionality  
- Increased `Prothrombin` time reflects impaired blood clotting caused by liver failure  
- Low `Platelets` are associated with advanced cirrhosis complications and portal hypertension  

---

## 🤖 Machine Learning Workflow

The project followed this workflow:

1. Data Cleaning  
2. Exploratory Data Analysis  
3. Missing Value Handling  
4. Feature Encoding & Scaling  
5. Train-Test Split  
6. Random Forest Modeling  
7. Hyperparameter Tuning using GridSearchCV  
8. SMOTE Oversampling for Class Imbalance  
9. Model Evaluation  
10. Permutation Importance Analysis  
11. Explanatory Visualizations  

---

## 🌲 Model Used

### Random Forest Classifier

Random Forest was selected because it:

- Handles mixed feature types effectively  
- Performs strongly on tabular healthcare datasets  
- Captures nonlinear feature interactions  
- Is robust against noise and outliers  

---

## 📈 Model Performance

### 🔻 Initial Model

The default Random Forest model achieved:

- Training Accuracy = 100%
- Poor test generalization

➡️ Severe overfitting detected.

---

### 🔻 After Hyperparameter Tuning

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

### 🔻 After Applying SMOTE

SMOTE significantly improved minority class learning and produced more balanced predictions.

### 📊 Final Test Performance

| Metric | Value |
|------|------|
| Accuracy | 0.76 |
| Macro F1-Score | 0.71 |
| Weighted F1-Score | 0.76 |

### 📌 Key Improvements

- Better prediction of transplant cases (`CL`)
- Improved balance across all target classes
- Reduced model bias toward majority outcomes
- More stable train-test generalization

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

These features are medically meaningful and strongly associated with liver disease severity and survival outcomes.

### 📷 Feature Importance Visualization

<img width="766" height="565" alt="image" src="https://github.com/user-attachments/assets/04d150e2-09e0-48ac-beb0-4a0c6bb4b780" />


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

- Liver-function biomarkers strongly influence cirrhosis outcome prediction  
- Class imbalance significantly affects healthcare machine learning models  
- SMOTE successfully improved minority class detection  
- Bilirubin and Prothrombin emerged as the strongest clinical indicators  
- The model learned medically meaningful patterns rather than random relationships  

---

## 📌 Conclusion

A complete machine learning pipeline was developed to predict cirrhosis patient outcomes using clinical and laboratory data.

The project successfully improved prediction quality through:

- Careful preprocessing  
- Hyperparameter tuning  
- SMOTE oversampling  
- Feature importance analysis  
- Clinically interpretable visualizations  

The final model achieved balanced multi-class performance while identifying medically meaningful predictors associated with liver disease severity and survival.

This project demonstrates how machine learning can support healthcare analytics and assist medical professionals in understanding important disease progression indicators.

---

## 👩‍💻 Author

Maryam Skaik  
Computer Science Graduate | Teaching Assistant | Data Science Enthusiast
