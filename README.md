# Insurance Charges Data Analysis

##  Project Overview

This project focuses on analyzing an **insurance dataset** to understand the factors that are associated with medical insurance charges.

The project starts with loading and exploring the raw dataset, followed by data cleaning, exploratory data analysis (EDA), feature engineering, encoding categorical variables, scaling numerical features, and statistical analysis.

The main goal is to identify which customer characteristics have a stronger relationship with insurance charges and prepare a clean dataset that can be used for further machine learning or predictive analysis.

---

##  Objectives

The main objectives of this project are:

* Understand the structure of the insurance dataset.
* Perform basic exploratory data analysis.
* Identify missing values and duplicate records.
* Analyze numerical and categorical variables.
* Convert categorical variables into numerical form.
* Create useful features from existing variables.
* Scale numerical features.
* Study the relationship between different features and insurance charges.
* Use Pearson correlation to measure relationships between numerical features and charges.
* Use Chi-square statistical testing to evaluate categorical features.
* Select relevant features for further analysis or machine learning.

---

##  Dataset

The project uses the `insurance.csv` dataset.

The dataset contains **1,338 records and 7 columns**.

### Dataset Features

| Column     | Description                        | Data Type   |
| ---------- | ---------------------------------- | ----------- |
| `age`      | Age of the individual              | Integer     |
| `sex`      | Gender of the individual           | Categorical |
| `bmi`      | Body Mass Index                    | Float       |
| `children` | Number of children/dependents      | Integer     |
| `smoker`   | Whether the individual is a smoker | Categorical |
| `region`   | Residential region                 | Categorical |
| `charges`  | Medical insurance charges          | Float       |

### Target Variable

The main variable of interest is:

`charges`

It represents the medical insurance charges associated with each individual.

---

##  Technologies Used

### Programming Language

* Python

### Libraries

* **Pandas** – Data loading, cleaning and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualizations
* **Scikit-learn** – Feature scaling
* **SciPy** – Statistical analysis

---

##  Project Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Categorical Encoding
   ↓
Feature Engineering
   ↓
Feature Scaling
   ↓
Correlation Analysis
   ↓
Chi-Square Feature Testing
   ↓
Feature Selection
   ↓
Final Dataset
```

---

# 1. Importing Required Libraries

The project starts by importing the required Python libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
```

These libraries are used for data manipulation, numerical calculations, visualization, and analysis.

---

# 2. Loading the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("insurance.csv")
```

Pandas makes it easy to read the CSV file and work with the data as a DataFrame.

---

# 3. Understanding the Dataset

Several basic operations are performed to understand the dataset.

### First few records

```python
df.head()
```

### Dataset information

```python
df.info()
```

### Statistical summary

```python
df.describe()
```

### Dataset dimensions

```python
df.shape
```

### Column names

```python
df.columns
```

These steps help understand the size, structure, data types, and basic statistics of the dataset before performing any transformations.

---

# 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the distribution of different variables and identify possible patterns.

## Numerical Features

The following numerical columns are analyzed:

```python
numeric_cols = ['age', 'bmi', 'children', 'charges']
```

Histograms with KDE are created to understand their distributions.

```python
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.show()
```

Boxplots are also created to identify the spread and possible outliers.

```python
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.show()
```

---

# 5. Categorical Data Analysis

Count plots are used to understand the distribution of categorical variables.

For example:

```python
sns.countplot(x=df['sex'])
```

```python
sns.countplot(x=df['children'])
```

```python
sns.countplot(x=df['smoker'])
```

These visualizations help understand how the observations are distributed across different categories.

---

# 6. Correlation Analysis

A correlation heatmap is created for numerical variables.

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
```

Correlation helps understand the strength and direction of relationships between numerical variables.

A positive correlation means that two variables tend to increase together, while a negative correlation means that one tends to decrease as the other increases.

---

# 7. Data Cleaning

A copy of the original dataset is created before performing transformations.

```python
df_cleaned = df.copy()
```

### Removing Duplicate Records

```python
df_cleaned.drop_duplicates(inplace=True)
```

### Checking Missing Values

```python
df_cleaned.isnull().sum()
```

### Checking Data Types

```python
df_cleaned.dtypes
```

This ensures that the dataset is clean and ready for further processing.

---

# 8. Encoding Categorical Variables

Machine learning and statistical analysis often require categorical values to be represented numerically.

## Gender Encoding

The `sex` column is converted into a binary numerical variable.

```python
df_cleaned['sex'] = df_cleaned['sex'].map({
    'male': 0,
    'female': 1
})
```

The column is then renamed:

```python
df_cleaned.rename(
    columns={'sex': 'is_female'},
    inplace=True
)
```

Therefore:

```text
male   → 0
female → 1
```

---

## Smoker Encoding

The `smoker` column is also converted into numerical form.

```python
df_cleaned['smoker'] = df_cleaned['smoker'].map({
    'no': 0,
    'yes': 1
})
```

It is renamed as:

```python
df_cleaned.rename(
    columns={'smoker': 'is_smoker'},
    inplace=True
)
```

Therefore:

```text
no  → 0
yes → 1
```

---

# 9. One-Hot Encoding Region

The `region` column contains multiple categories.

Instead of assigning arbitrary numerical values to these categories, one-hot encoding is used.

```python
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['region'],
    drop_first=True
)
```

This creates columns such as:

```text
region_northwest
region_southeast
region_southwest
```

`drop_first=True` removes one category to avoid unnecessary redundancy.

---

# 10. BMI Feature Engineering

BMI is converted into meaningful categories.

```python
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=[
        'Underweight',
        'Normal weight',
        'Overweight',
        'Obese'
    ]
)
```

The categories are:

| BMI Range   | Category      |
| ----------- | ------------- |
| Below 18.5  | Underweight   |
| 18.5 – 24.9 | Normal weight |
| 25 – 29.9   | Overweight    |
| 30+         | Obese         |

The new categorical feature is then converted using one-hot encoding.

```python
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['bmi_category'],
    drop_first=True
)
```

This allows BMI categories to be used in statistical analysis.

---

# 11. Feature Scaling

The numerical features are standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']

scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(
    df_cleaned[cols]
)
```

Standardization transforms the features so that they have a comparable scale.

This is particularly useful when numerical variables have different ranges.

---

# 12. Converting Boolean Columns

One-hot encoded columns can be stored as Boolean values.

They are converted into `0` and `1`.

```python
bool_cols = df_cleaned.select_dtypes(
    include='bool'
).columns

df_cleaned[bool_cols] = df_cleaned[
    bool_cols
].astype(int)
```

This makes the final dataset easier to use for statistical analysis and machine learning.

---

# 13. Pearson Correlation Analysis

Pearson correlation is used to measure the relationship between selected features and insurance charges.

```python
from scipy.stats import pearsonr
```

The selected features include:

* Age
* BMI
* Number of children
* Gender
* Smoking status
* Region
* BMI categories

The correlation is calculated using:

```python
pearsonr(feature, charges)
```

The results are stored in a DataFrame and sorted by correlation.

This helps identify features that have stronger linear relationships with insurance charges.

---

# 14. Chi-Square Statistical Test

The project also uses the **Chi-square test of independence** for categorical features.

```python
from scipy.stats import chi2_contingency
```

First, insurance charges are divided into four groups using quartiles:

```python
df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)
```

The categorical features are then tested against these charge groups.

The significance level used in the project is:

```python
alpha = 0.05
```

The decision is based on the p-value:

```text
p-value < 0.05
    → Reject Null Hypothesis
    → Feature is considered significant

p-value >= 0.05
    → Do not reject Null Hypothesis
    → Feature is considered less significant
```

This provides a statistical way of evaluating whether categorical features have an association with different insurance charge groups.

---

# 15. Feature Selection

After performing correlation and Chi-square analysis, selected features are retained for the final dataset.

The project creates the following final DataFrame:

```python
final_df = df_cleaned[
    [
        'age',
        'is_female',
        'bmi',
        'children',
        'is_smoker',
        'charges',
        'region_southeast',
        'bmi_category_Obese'
    ]
]
```

The final dataset contains the selected features that can be used for further analysis or predictive modeling.

---

#  Key Analysis Performed

The project covers several important data analytics concepts:

### Data Understanding

* Dataset shape
* Column names
* Data types
* Descriptive statistics

### Data Cleaning

* Duplicate removal
* Missing-value checking
* Data type checking

### Exploratory Data Analysis

* Histograms
* KDE plots
* Boxplots
* Count plots
* Correlation heatmap

### Feature Engineering

* BMI categories
* Binary variables
* One-hot encoded variables

### Feature Transformation

* Standardization
* Boolean-to-integer conversion

### Statistical Analysis

* Pearson correlation
* Chi-square test of independence
* p-value based feature evaluation

---

#  Project Structure

A recommended GitHub structure for this project is:

```text
insurance-charges-analysis/
│
├── insurance.csv
├── insurance.ipynb
├── README.md
└── requirements.txt
```

### File Description

| File               | Description                                       |
| ------------------ | ------------------------------------------------- |
| `insurance.csv`    | Original insurance dataset                        |
| `insurance.ipynb`  | Jupyter Notebook containing the complete analysis |
| `README.md`        | Project documentation                             |
| `requirements.txt` | Required Python libraries                         |

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/insurance-charges-analysis.git
```

Move into the project directory:

```bash
cd insurance-charges-analysis
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

---

## 3. Install Required Libraries

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

Or create a `requirements.txt` file and install everything using:

```bash
pip install -r requirements.txt
```

---

## 4. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
insurance.ipynb
```

Run the cells from top to bottom.

---

#  Requirements

The main libraries used in this project are:

```text
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
jupyter
```

---

# What I Learned From This Project

This project helped me understand the complete workflow of a basic data analysis project.

Some of the important concepts practiced include:

* Loading data using Pandas
* Understanding a new dataset
* Performing EDA
* Working with numerical and categorical variables
* Removing duplicate records
* Checking missing values
* Encoding categorical variables
* One-hot encoding
* Feature engineering
* Feature scaling
* Correlation analysis
* Pearson correlation
* Hypothesis testing
* Chi-square test
* p-values and significance levels
* Feature selection

---

#  Future Improvements

The current project focuses mainly on data analysis and feature selection. It can be extended further by building a complete insurance cost prediction system.

Possible future improvements include:

* Build Linear Regression models
* Try Random Forest Regression
* Try Gradient Boosting models
* Compare different machine learning models
* Evaluate models using MAE, MSE and RMSE
* Perform train-test splitting
* Perform cross-validation
* Tune model hyperparameters
* Create additional visualizations
* Build an interactive Power BI dashboard
* Deploy the final prediction model using Streamlit or Flask

---

#  Conclusion

This project demonstrates a complete beginner-to-intermediate data analysis workflow using an insurance dataset.

Starting from the raw data, the project performs data exploration, cleaning, feature engineering, encoding, scaling, correlation analysis, and statistical feature selection.

The final processed dataset provides a suitable foundation for building a machine learning model to predict or further analyze insurance charges.

---

##  Author

**Divyansh Sharma**

---

##  If you found this project useful

Feel free to  star the repository and explore the notebook for the complete implementation.
