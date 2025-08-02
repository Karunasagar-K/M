# 🧠 AutoML Streamlit App

An interactive AutoML application built with **Streamlit** that allows users to:
- Upload datasets (CSV)
- Run **Exploratory Data Analysis (EDA)** using **Sweetviz** and **YData-Profiling**
- Automatically detect and perform **Classification** or **Regression**
- Handle **imbalanced datasets** using **SMOTE** or **Random Undersampling**
- Compare model performance with standard evaluation metrics
- Visualize results in an interactive and user-friendly interface

---

## 🚀 Features

### ✅ Dataset Upload
- Users can upload `.csv` files through Streamlit UI.
- Automatically loads the dataset and displays sample rows and info.

### 📊 Exploratory Data Analysis (EDA)
- Choose between **Sweetviz** or **YData Profiling** to perform detailed data profiling.
- Generates HTML-based interactive reports embedded inside the app.

### 🔍 Target Column Selection
- App automatically detects column types (numerical or categorical).
- You can select the target column (y) for model training.

### 🧮 Model Type Detection
- Based on the data type of target column:
  - If numeric → Regression
  - If categorical → Classification

### ⚖️ Imbalanced Dataset Handling
- Option to apply:
  - **SMOTE (Synthetic Minority Oversampling Technique)**
  - **Random Undersampling**
- Only applicable for classification tasks.

### 🛠️ Model Selection and Training
- For **Classification**:
  - Logistic Regression, Random Forest, XGBoost, etc.
- For **Regression**:
  - Linear Regression, Random Forest Regressor, XGBoost Regressor, etc.

### 📈 Evaluation Metrics
- **Classification**:
  - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression**:
  - RMSE, MAE, R² Score

### 📊 Result Visualization
- Interactive charts and comparison of model scores.
- Download trained model and reports (optional extensions).

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – App framework
- **Pandas, NumPy** – Data handling
- **Scikit-learn** – ML models & metrics
- **XGBoost, LightGBM** – Advanced ML models
- **Sweetviz, YData Profiling** – EDA
- **Imbalanced-learn** – SMOTE, undersampling

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/automl-streamlit-app.git
   cd automl-streamlit-app
2.Install Dependencies:   
   pip install -r requirements.txt
3.Run the app
streamlit run app.py
**Folder Structure**
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── reports/               # EDA reports (optional)

✨ Future Enhancements
Add model explainability with SHAP/LIME

Allow feature engineering customization

Model export/download option

👨‍💻 Author
Karunasagar K
Feel free to connect on GitHub | LinkedIn!








