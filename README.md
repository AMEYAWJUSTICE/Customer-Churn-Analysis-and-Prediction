

````markdown
# 📊 Telco Customer Churn Prediction Using Machine Learning

##  Project Overview

This project focuses on building and evaluating machine learning models to predict customer churn in a telecommunications company. The objective is to identify customers likely to discontinue service and understand the key factors influencing churn behavior.

By leveraging predictive analytics, businesses can improve customer retention strategies, reduce revenue loss, and enhance service quality.

---

##  Problem Statement

Telecommunication companies experience significant revenue loss due to customer churn. The challenge is to develop a predictive system that can:

- Identify customers likely to churn
- Understand key drivers of churn
- Support data-driven retention strategies

---

##  Dataset Description

The dataset used is the **Telco Customer Churn Dataset**, which contains customer demographics, account information, and subscribed services.

### Key Features:

- `customerID` – Unique identifier (dropped during preprocessing)
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `InternetService`
- `Contract`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn` (Target Variable)

---

##  Project Workflow

### 1. Data Loading & Exploration

Initial exploration was performed using Pandas:

- `df.head()`
- `df.info()`
- `df.shape`
- `df.describe()`
- `df.isnull().sum()`

A file path error was initially encountered and resolved.

---

### 2. Data Cleaning & Preprocessing

Key preprocessing steps included:

- Dropping irrelevant column: `customerID`
- Converting `TotalCharges` from object → numeric
- Handling missing values (filled with 0)
- Fixing incorrect/randomized `Churn` values by reloading correct dataset
- Encoding categorical variables:
  - Binary encoding (0/1)
  - One-hot encoding for multi-class features

---

### 3. Feature Engineering

- Selected important predictive variables
- Removed redundant or highly correlated features
- Ensured dataset was fully numeric for ML models

---

### 4. Data Visualization (EDA)

Exploratory visualizations included:

- Gender distribution
- Contract type distribution
- Monthly charges distribution
- Churn distribution analysis

Libraries used:
- Matplotlib
- Seaborn

---

### 5. Data Splitting

Dataset was split into:

- **Training set (80%)**
- **Testing set (20%)**

Using:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
````

---

##  Machine Learning Models Used

Three classification models were trained and evaluated:

### 1. Logistic Regression

* Accuracy: **82.04%**
* ROC-AUC: **0.8625**
* Best performing model overall

### 2. Random Forest Classifier

* Accuracy: **78.64%**
* Lower recall for churn class

### 3. XGBoost Classifier

* Accuracy: **79.13%**
* Balanced performance but slightly lower than Logistic Regression

---

##  Best Model Selection

Based on evaluation metrics:

✔ Logistic Regression was selected as the best model due to:

* Highest accuracy
* Strong ROC-AUC score
* Better balance between precision and recall

---

##  Evaluation Metrics Used

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC Curve

---

##  Model Export

Trained models were saved using `joblib` for deployment:

```
logistic_regression_model.joblib
random_forest_model.joblib
xgboost_model.joblib
```

---

##  Technologies Used

* Python 
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib

---

## 📈Key Insights

* Customers with shorter tenure are more likely to churn
* Month-to-month contracts show higher churn rates
* Higher monthly charges correlate with increased churn risk
* Long-term contracts reduce churn probability

---

## Deployment Potential

This project can be extended into a full web application using:

* Streamlit
* Flask API
* FastAPI

Example deployment tools:

* Streamlit Cloud
* Render
* Railway

---

##  Project Structure

```
Telco-Customer-Churn-Prediction/
│
├── data/
│   └── Telco_Customer_Churn_Dataset.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── models/
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   └── xgboost_model.joblib
│
├── app/
│   └── streamlit_app.py
│
├── README.md
└── requirements.txt
```

---

##  Installation

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
```

---

##  Run Project

### Run Jupyter Notebook:

```bash
jupyter notebook
```

### Run Streamlit App:

```bash
streamlit run app/streamlit_app.py
```

---

##  Future Improvements

* Improve model using deep learning
* Handle class imbalance using SMOTE
* Build real-time prediction API
* Add SHAP explainability for interpretability
* Deploy as SaaS churn prediction tool

---

##  Author

**AMEYAW ASANTE JUSTICE**
IT | Machine Learning | Data Science | Digital Systems

---

## License

This project is for educational and research purposes.

---

##  Acknowledgements

* IBM Telco Dataset
* Scikit-learn Documentation
* Kaggle Community

```

