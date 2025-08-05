# 🧠 AutoML Streamlit App

An interactive AutoML application built with **Streamlit** that allows users to:

* Upload datasets (CSV)
* Run **Exploratory Data Analysis (EDA)** using **Sweetviz** and **YData-Profiling**
* Automatically detect and perform **Classification** or **Regression**
* Handle **imbalanced datasets** using **SMOTE** or **Random Undersampling**
* Compare model performance with standard evaluation metrics
* Visualize results in an interactive and user-friendly interface

---

## 🚀 Features

### ✅ Dataset Upload

* Users can upload `.csv` files through Streamlit UI.
* Automatically loads the dataset and displays sample rows and info.
* Related function in code: `data_collection()`

### 📊 Exploratory Data Analysis (EDA)

* Choose between **Sweetviz** or **YData Profiling** to perform detailed data profiling.
* Generates HTML-based interactive reports embedded inside the app.
* Visuals include:

  * Pairplot
  * Boxplot
  * Correlation heatmap
* Related function in code: `data_understanding(df)`

### 🔍 Target Column Selection

* App automatically detects column types (numerical or categorical).
* User can select target variable (y) and features (X).
* Related function in code: `Feature_selection(df)`

### 🧮 Model Type Detection

* Based on the data type of the target column:

  * If numeric → Regression
  * If categorical → Classification
* Automatically detected in `main()`

### ⚖️ Imbalanced Dataset Handling

* Optionally apply:

  * **SMOTE (oversampling)**
  * **Random Undersampling**
* Only for classification tasks
* Related functions/libraries used:

  * `SMOTE` from `imblearn`
  * `RandomUnderSampler`
  * Balance check via `check_class_balance(Target)`

### 🛠️ Model Selection and Training

* For **Classification**:

  * Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN
  * Related function: `perform_classification(...)`

* For **Regression**:

  * Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN
  * Includes Polynomial and OLS (for single feature)
  * Related function: `perform_regression(...)`

### 📈 Evaluation Metrics

* **Classification**:

  * Accuracy, Precision, Recall, F1-Score, Confusion Matrix
* **Regression**:

  * R² Score, RMSE, MAE

### 📊 Result Visualization

* Display evaluation metrics in table format
* Identify and highlight best performing model
* Interactive result viewing with Streamlit widgets

### 📂 Model Prediction

* User inputs feature values in UI
* Predicts target value using best trained model
* Handled within `main()` after model training

---

## 💪 Data Processing Pipeline

### 📊 Data Understanding

* Shapes, types, missing values, statistics, and visual distributions

### ⚖️ Data Preprocessing

* Handle missing values using:

  * Mean (numeric), Mode (categorical), Interpolation (datetime)
* Detect and treat outliers using IQR method
* Related function: `data_preprocessing(df)`

### ⚖️ Data Preparation

* Label encoding for categorical
* Feature engineering from datetime
* StandardScaler or MinMaxScaler based on Gaussian check
* Related function: `data_preparation(df)`

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – App framework
* **Pandas, NumPy** – Data handling
* **Scikit-learn** – ML models & metrics
* **XGBoost, LightGBM** – Advanced models
* **Sweetviz, YData Profiling** – EDA tools
* **Imbalanced-learn** – SMOTE, undersampling

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/automl-streamlit-app.git
   cd automl-streamlit-app
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run app.py
   ```

### 📂 Folder Structure

```
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── reports/               # EDA reports (optional)
```

---

## ✨ Future Enhancements

* Model explainability with SHAP/LIME
* Feature engineering customization
* Model export/download option

---

## 👨‍💼 Author

**Karunasagar K**

Feel free to connect on GitHub | LinkedIn!
