# AI-Health-Data-Science-Agent



# Heart Disease Prediction using Data Science

## Project Overview

This project uses **data science and machine learning** techniques to predict whether a patient is likely to have heart disease based on medical attributes. The workflow includes **data preprocessing, exploratory analysis, feature scaling, model training, and evaluation**.

The model is trained using **Logistic Regression** and evaluated with metrics such as **accuracy, classification report, and confusion matrix**.

---

# Dataset

The dataset used in this project is the **Cleveland Heart Disease dataset**.

It contains medical attributes such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise induced angina
* ST depression
* Slope
* Number of major vessels
* Thalassemia
* Target (heart disease presence)

The **target variable** is converted into a binary classification:

* `1` → Heart Disease Present
* `0` → No Heart Disease

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Project Workflow

## 1 Data Loading

The dataset is loaded using **Pandas**.

```python
df = pd.read_csv("processed.cleveland.csv", header=None)
```

Column names are then assigned to the dataset.

---

# 2 Data Exploration

Basic analysis is performed:

* Dataset shape
* Data types
* Summary statistics
* Missing values

```python
df.info()
df.describe()
df.isnull().sum()
```

---

# 3 Data Cleaning

The dataset contains missing values represented by `"?"`.

Steps performed:

* Replace `"?"` with `NaN`
* Convert columns to numeric
* Fill missing values using **median**

```python
df.replace("?", np.nan, inplace=True)
df.fillna(df.median(), inplace=True)
```

---

# 4 Data Preprocessing

Additional preprocessing steps include:

### Convert Target Variable

Convert multi-class target into **binary classification**.

```python
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
```

### Remove Duplicates

```python
df.drop_duplicates(inplace=True)
```

### Handle Outliers

Outliers are clipped using the **1st and 99th percentile**.

```python
df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
```

---

# 5 Feature Scaling

Standardization is applied to normalize the features.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 6 Train Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

# 7 Model Training

The model used in this project is **Logistic Regression**.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

# 8 Model Evaluation

Model performance is evaluated using:

* Accuracy score
* Classification report
* Confusion matrix

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Confusion matrix is visualized using **Seaborn heatmap**.

---

# Healthcare Prediction Agent

A simple **healthcare prediction function** is created to test new patient data.

Example:

```python
patient = [55,1,3,130,250,0,1,150,0,1.5,2,0,3]
healthcare_agent(patient)
```

The model processes the patient data and predicts whether heart disease is present.

---

# Output

The project produces:

* Data analysis summary
* Visualization of outliers
* Confusion matrix
* Prediction results for new patients

---

# How to Run the Project

1. Install required libraries

pip install pandas numpy matplotlib seaborn scikit-learn

2. Open the Jupyter Notebook

jupyter notebook datascience.ipynb

3. Run all cells sequentially.
