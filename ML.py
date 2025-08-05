import streamlit as st                      #For Streamlit
import pandas as pd                         #For Data manipulation 
import matplotlib.pyplot as plt             #For visualization
import seaborn as sns                       # ,,   ,,
import sweetviz as sv
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.impute import SimpleImputer    #For Imputing
import numpy as np                          #For Numerical operation
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import normaltest
from sklearn.model_selection import train_test_split,cross_val_predict  #For Splitting Train Test Data
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def data_collection():   
    uploaded_file = st.file_uploader("Upload file", type=None)
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}. Please upload a valid file.")
            return
        st.write(df.head(10))
        
        return df

def Feature_selection(df):
        # Allow user to select features (X) and target variable (y)
        Feature = st.multiselect("Select features for X", df.columns)
        Target = st.selectbox("Select target variable (y)", df.columns)

        return Feature,Target

def data_understanding(df):
    st.subheader("Data Overview")
    st.write("Data shape:", df.shape)
    st.write("Data types:", df.dtypes)
    st.write("Missing values:", df.isna().sum())
    st.write("Summary statistics:", df.describe())

    #EDA
    # Visualize distributions of numerical variables
    st.subheader("Distributions of numerical variables:")
    pairplot_fig = sns.pairplot(df)
    st.pyplot(pairplot_fig.figure,dpi=500)  # Access the underlying matplotlib figure

    st.subheader("Distribution of box plots:")
    fig1,ax1 = plt.subplots()
    sns.boxplot(df,ax=ax1)
    st.pyplot(fig1, dpi=500)

    # Visualize correlations among features
    st.subheader("Correlation heatmap:")
    fig2,ax2 = plt.subplots(figsize=(10, 8), dpi=500)
    sns.heatmap(df.corr(numeric_only=True), annot=True,ax=ax2)
    st.pyplot(fig2)

    profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)  #Adjust if necessary
    profile_html = profile.to_html()
    st.components.v1.html(profile_html, height=5000)


    #a = sv.analyze(df)
     # Add a button to show the Sweetviz HTML report
    #if st.button("Show Sweetviz HTML Report"):
       # a.show_html('a.html')
        #with open('a.html', 'r') as file:
         #   html_content = file.read()
       # components.html(html_content, height=3000)
    
    return df

def data_preprocessing(df):
    #Diagnosis - Checking for Null values
    missing_values = df.isna().sum()
    st.subheader("Missing values in the dataset:")
    st.write(missing_values)
    
    if df.isnull().values.any():
        st.write("Imputing missing values...")

        # Initialize a dictionary to store imputation strategies for each column
        imputation_strategies = {}

        # Identify column types and assign imputation strategies accordingly
        for column in df.columns:
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                # Numerical columns: Impute missing values with mean
                imputation_strategies[column] = 'mean'
            elif df[column].dtype == 'object':
                # Categorical columns: Impute missing values with mode
                imputation_strategies[column] = 'most_frequent'
            elif df[column].dtype == 'datetime64[ns]':
                # Datetime columns: Interpolate missing values
                imputation_strategies[column] = 'interpolate'

        # Apply imputation strategies
        for column, strategy in imputation_strategies.items():
            imputer = SimpleImputer(strategy=strategy)
            df[column] = imputer.fit_transform(df[[column]])[:,0]

        st.write("Missing values imputed using custom strategies.")
    else:
        st.write("No missing values detected.")

    # Diagnosis - Checking for Outliers
    def detect_outliers_iqr(series, threshold=1.5):

        # Check if the series contains only 0s and 1s
        if set(series) == {0, 1}:
            return []  # Return an empty list if the series contains only 0s and 1s

        quartile_1, quartile_3 = np.percentile(series, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (threshold * iqr)
        upper_bound = quartile_3 + (threshold * iqr)
        outliers = np.where((series < lower_bound) | (series > upper_bound))[0]
        return outliers

    # Detect and treat outliers for each numerical column
    numerical_columns = df.select_dtypes(include=np.number).columns
    outliers_detected = False
    for column in numerical_columns:
        outliers = detect_outliers_iqr(df[column])
        if len(outliers) > 0:
            outliers_detected = True
            st.write("Outliers detected in column:", column)
            st.write("Outlier indices:", outliers)
            st.write("Outliers:", df[column].iloc[outliers])
            # Treat outliers by replacing with median
            median = np.median(df[column])
            df[column].iloc[outliers] = median
            st.write("Outliers treated with median value:", median)
        else:
            st.write("No outliers detected in column:", column)

    if outliers_detected:
        st.write("Data after treating outliers:")
        st.write(df)
    else:
        st.write("No outliers detected. No treatment performed.")

    return df

def data_preparation(df):
    """
    Preprocess the data including label encoding for categorical columns and datetime column conversion.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: Preprocessed DataFrame.
    """
    # Label encoding for categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = pd.factorize(df[col])[0]

    # Convert datetime column
    if  all(df[col].dtype == 'datetime64[ns]' for col in df.columns):
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
        df['year'] = df['Datetime'].dt.year
        df['month'] = df['Datetime'].dt.month
        df['day'] = df['Datetime'].dt.day
        df['dayofweek_num'] = df['Datetime'].dt.dayofweek
        df['Hour'] = df['Datetime'].dt.hour
        df['minute'] = df['Datetime'].dt.minute
        
    # Check if data follows Gaussian distribution
    for column in df.columns:
        if column != 'Target':
    # Check if data follows Gaussian distribution
            p_value = normaltest(df[column])[1]
            if p_value < 0.05:
    # Data doesn't follow Gaussian distribution, so use Min-Max scaling
                scaler = MinMaxScaler()
                df[column] = scaler.fit_transform(df[[column]])
                print(f"Min-Max Scaled {column}")
            else:
    # Data follows Gaussian distribution, so use Standardization
                scaler = StandardScaler()
                df[column] = scaler.fit_transform(df[[column]])
                print(f"Standardized {column}")
            


    return df

def check_class_balance(Target):
    """
    Check if the target variable is balanced or not.

    Parameters:
    y (Series): Target variable.

    Returns:
    bool: True if the target variable is imbalanced, False otherwise.
    """
    class_counts = Target.value_counts()
    majority_class_count = class_counts.max()
    minority_class_count = class_counts.min()
    imbalance_ratio = majority_class_count / minority_class_count
    return imbalance_ratio != 1.0

def perform_classification(X_train, X_test, y_train, y_test):
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Random Forest Classifier': RandomForestClassifier(),
        'Gradient Boosting Classifier': GradientBoostingClassifier(),
        'XGBoost Classifier': XGBClassifier(),
        'K-Nearest Neighbors Classifier': KNeighborsClassifier()
    }

    results = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    for clf_name, clf in classifiers.items():

        # Train the model
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Append results to dataframe
        results.loc[len(results)] = [clf_name, accuracy, precision, recall, f1_score]

    st.write("Classification Results:")
    st.write(results.pivot_table(index='Algorithm', aggfunc='mean'))

    # Select the best model based on the highest F1-score
    best_model = results.loc[results['F1-Score'].idxmax()]

    st.write("\nBest Model:")
    st.write(best_model)

    # Train the best model on the entire dataset
    best_model_name = best_model['Algorithm']
    best_model_classifier = classifiers[best_model_name]
    best_model_classifier.fit(X_train, y_train)

    # Store the best model classifier in session state
    st.session_state.classifiers = {best_model_name: best_model_classifier}

    return best_model_classifier,best_model_name

def perform_regression(X_train, y_train, X_test, y_test):
    """
    Perform regression on the dataset.

    Parameters:
    X_train (DataFrame): Training features.
    y_train (DataFrame): Training target.
    X_test (DataFrame): Test features.
    y_test (DataFrame): Test target.

    Returns:
    DataFrame: Results of different regression algorithms.
    """
    algorithms = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net Regression': ElasticNet(alpha=0.1),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting Regressor': GradientBoostingRegressor(),
        'XGBoost Regressor': XGBRegressor(),
        'KNN Regressor': KNeighborsRegressor()
    }

    results = []
    for name, model in algorithms.items():
        if isinstance(model, PolynomialFeatures):
            continue  # Skip polynomial features
        else:
            y_pred = cross_val_predict(model, X_train, y_train, cv=5)
            r2 = r2_score(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            results.append({'Algorithm': name, 'R2 Score': r2, 'RMSE': rmse})
            results.append({'Algorithm': name, 'R2 Score': r2, 'RMSE': rmse})

    # Polynomial Regression
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X_train)
    poly_model = LinearRegression()
    y_pred_poly = cross_val_predict(poly_model,X_poly,y_train,cv=5)
    r2_poly = r2_score(y_train, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_pred_poly))
    results.append({'Algorithm': 'Polynomial Regression', 'R2 Score': r2_poly, 'RMSE': rmse_poly})

# OLS Regression if only one feature
    if X_train.shape[1] == 1:
        X_train = sm.add_constant(X_train)  # Add constant for intercept
        ols_model = sm.OLS(y_train, X_train)
        ols_results = ols_model.fit()
        r2_ols = r2_score(y_test, ols_results.predict(sm.add_constant(X_test)))
        rmse_ols = np.sqrt(mean_squared_error(y_test, ols_results.predict(sm.add_constant(X_test))))
        results.append({'Algorithm': 'OLS Regression', 'R2 Score': r2_ols, 'RMSE': rmse_ols})

    results_df = pd.DataFrame(results)
    pivot_table = results_df.pivot_table(index='Algorithm', values=['R2 Score', 'RMSE'], aggfunc=np.mean)

    # Get the best algorithm based on R2 score
    best_algorithm = pivot_table['R2 Score'].idxmax()
    best_r2 = pivot_table.loc[best_algorithm, 'R2 Score']
    best_rmse = pivot_table.loc[best_algorithm, 'RMSE']

    return pivot_table, best_algorithm, best_r2

def main():
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Automated Machine Learning Framework")
    st.write("This app automatically selects between regression and classification based on the target column's data type.")

    # Data Collection
    df = data_collection()

    if df is not None:
        # Feature Selection
        Feature, Target = Feature_selection(df)

        # Data Understanding
        df = data_understanding(df)

        # Data Preprocessing
        df = data_preprocessing(df)

        # Data Preparation
        df = data_preparation(df)
        st.write(df)

        # Check class balance
        is_imbalanced = check_class_balance(df[Target])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[Feature], df[Target], test_size=0.2, random_state=42)

        if Feature and Target:
            if set(df[Target]) == {0, 1} or set(df[Target]) =={0, 1 ,2}:
                st.write("Target column has only 0s and 1s. Treating it as categorical for classification.")
                st.write("Performing Classification")
                if is_imbalanced:
                    st.write("Target column is imbalanced. Performing resampling...")
                # Oversample or undersample the data
                    if len(df[Target]) < 100:
                        st.write("Applying SMOTE (oversampling) for balancing the data")
                        smote = SMOTE()
                        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    else:
                        st.write("Applying RandomUnderSampler (undersampling) for balancing the data")
                        rus = RandomUnderSampler(sampling_strategy='auto')
                        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
                    # Perform classification on resampled data
                    best_model_classifier,best_model_name = perform_classification(X_train_resampled, X_test, y_train_resampled, y_test)
                else:
                    st.write("Target column is balanced. Performing classification...")
                    # Perform classification on original data
                    best_model_classifier,best_model_name = perform_classification(X_train, X_test, y_train, y_test)

                
                # Allow user to input values for selected features and predict output
                st.subheader("Predict Output:")
                input_values = {}
                for feature in Feature:
                    input_values[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                if st.button("Predict"):
                    input_data = pd.DataFrame([input_values])
                    prediction = None
                    
                    print("Best Model Name:", best_model_name)
                    print("Session State Classifiers:", st.session_state.classifiers)

                    if best_model_name is not None and "classifiers" in st.session_state:
                        best_model_classifier = st.session_state.classifiers.get(best_model_name)
                        if best_model_classifier is not None:
                        # Make prediction using the best model
                            prediction = best_model_classifier.predict(input_data)
                            st.write(f"Predicted {Target} value using {best_model_name}: {prediction[0]}")
                        else:
                            st.error("Best model classifier not found. Please train a model first.")
                    else:
                        st.error("Best model name is not initialized or classifiers have not been initialized.")



            elif np.issubdtype(df[Target].dtype, np.number):
                # Regression task
                st.write("Target column is numerical and continuous. Treating it as Regression.")
                st.write("Performing Regression...")
                pivot_table, best_algorithm, best_r2 = perform_regression(X_train, y_train, X_test, y_test)
                st.write("Regression Results:")
                st.write(pivot_table)
                st.write("Best Algorithm:", best_algorithm)
                st.write("R2 Score:", best_r2)

                # Allow user to input values for selected features and predict output
                st.subheader("Predict Output:")
                input_values = {}
                for feature in Feature:
                    input_values[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                if st.button("Predict"):
                    input_data = pd.DataFrame([input_values])
                    prediction = None

                    if "Linear Regression" in best_algorithm:
                        model = LinearRegression()
                    elif "Ridge Regression" in best_algorithm:
                        model = Ridge(alpha=1.0)
                    elif "Lasso Regression" in best_algorithm:
                        model = Lasso(alpha=0.1)
                    elif "Polynomial Regression" in best_algorithm:
                        model = PolynomialFeatures(degree=2)
                        X_poly = model.fit_transform(input_data)
                        model = LinearRegression()
                    elif "Decision Tree Regressor" in best_algorithm:
                        model = DecisionTreeRegressor()
                    elif "Random Forest Regressor" in best_algorithm:
                        model = RandomForestRegressor(n_estimators=100)
                    elif "XGBoost Regressor" in best_algorithm:
                        model = XGBRegressor()
                    elif "KNN Regressor" in best_algorithm:
                        model = KNeighborsRegressor()

                    model.fit(X_train, y_train)
                    if "Polynomial Regression" in best_algorithm:
                       model = PolynomialFeatures(degree=2)
                       X_poly = model.fit_transform(input_data)
                       model = LinearRegression()
                       prediction = model.predict(X_poly)
                    else:
                        prediction = model.predict(input_data)

                    st.write(f"Predicted {Target} value using {best_algorithm}: {prediction[0]}")

if __name__ == "__main__":
    main()