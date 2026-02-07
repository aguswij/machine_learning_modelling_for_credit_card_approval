import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from numpy.random import default_rng as rng
import datetime
import time

col_a1, col_a2 = st.columns([0.8, 0.2])
with col_a1:
    st.write("Final Project Data Science")
    st.markdown("**Agus Wijaya**")
    st.write("Student of Data Science and Analyst Bootcamp Dibimbing @2025")
with col_a2:
    st.image("Picture_AW.jpg")
st.write("_________________________________________________________")
col_b1, col_b2 = st.columns(2)
with col_b1:
    st.header("Machine Learning Modelling for")
    st.markdown("<h2 style='color:red;'>Credit Card Approval Prediction</h2>",unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("Classification Modelling with Python")
with col_b2:
    st.image("Credit_Card-1.jpeg")
    st.image("Credit_Card-2.jpeg")
st.write("_________________________________________________________")
st.markdown("**Project Overview :**")
st.markdown("""
            - This project will attempt to build a machine learning model (Classification) to predict whether a credit card applicant falls into the category of a good or bad client by analyzing the applicant's credit card payment transaction history
            - The analysis method that will be used is the CRISP-DM Framework (CRoss-Industry Standard Process for Data Mining) which includes the following steps : Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation and Deployment
""")
st.write("_________________________________________________________")
st.markdown("**Business Understanding :**")
st.markdown("""
            - **Background** : From the incoming data, it was detected that there are a total of 438.557 credit card applicants with the following data distribution : Male : 144.117 (33%), Female : 294.440 (67%), Commercial Associate : 100.757 (22,97%), Pensioner : 75,493 (17.2%), State Servant : 36,186 (8.2%), Student : 17 (0,004%), Working : 226.104 (51,5%)
            - **Objective** : Delivering prediction for a credit card applicant falls into the category of a good or bad client by analyzing the applicant's credit card payment transaction history so the bank can decide whether to issue a credit card to the applicant or not
            - **Problem** : In this project, a lot of data with missing values (around 30%) were found, so it is necessary to delete this data, which could potentially affect the quality of the data analysis. In addition, many outlier data, data without pairs and data imbalance were also found, which could potentially affect the quality of the data analysis
""")
st.write("_________________________________________________________")
st.markdown("**Data Understanding :**")
st.write("Dataset Name     : Credit Card Approval Prediction")
st.write("Dataset Source   : https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction")
st.image("Dataset-1.jpeg")
st.image("Dataset-2.jpeg",width=700)
st.write("_________________________________________________________")
st.markdown("**Data Preparation :**")
st.write("Exploratory Data Analysis of Numerical Data")
st.image("EDA-1.jpeg")
st.image("EDA-2.jpeg")
st.image("EDA-3.jpeg")
st.image("EDA-4.jpeg")
st.write("Exploratory Data Analysis of Categorical Data")
st.image("EDA-5.jpeg")
st.image("EDA-6.jpeg")
st.write("Correlation Heatmap of Numerical Data")
st.image("EDA-7.jpeg",width=700)
st.write("_________________________________________________________")
st.markdown("**Modelling :**")

# Library Preparation

# Basic Library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Cross-Validation
from sklearn.model_selection import train_test_split

# Model
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
from sklearn.svm import SVC

# Model Evaluation
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import loguniform, uniform, randint

# Import Dataset to Dataframe
df_app = pd.read_csv('application_record.csv', encoding = "ISO-8859-1")
df_credit = pd.read_csv('credit_record.csv', encoding = "ISO-8859-1")

# Data Preparation of Application Record Data
# Handling Missing Values
df_app = df_app.dropna()

# Feature Engineering of Credit Record Data
# Create Converter Value of Credit Status (Encoding)
def encoding_status(x):
    if x == 'C':
        return 8
    elif x == 'X':
        return 7
    elif x == '0':
        return 6
    elif x == '1':
        return 5
    elif x == '2':
        return 4
    elif x == '3':
        return 3
    elif x == '4':
        return 2
    else:
        return 1
df_credit['ENCODING_STATUS'] = df_credit['STATUS'].apply(encoding_status)


# Feature Engineering of Credit Record Data
# Grouping Customer Category by History Status
df_cc = df_credit.groupby('ID')['ENCODING_STATUS'].mean().round(1).sort_values(ascending=False).reset_index()
df_cc.rename(columns={'ENCODING_STATUS': 'RESUME_STATUS'}, inplace=True)

# Convert Data Type  Float to Integer and rounded
df_cc['RESUME_STATUS'] = df_cc['RESUME_STATUS'].round().astype(int)

# Feature Engineering of Customer Category Dataframe
# Encoding RESUME_STATUS
def encoding_resta(x):
    if x > 4:
        return 1
    else:
        return 0
df_cc['CUSTOMER_CATEGORY'] = df_cc['RESUME_STATUS'].apply(encoding_resta)

# Merging Application Record Data and Customer Category Data
df_gabung = pd.merge(df_app,df_cc, on = 'ID', how = 'inner')

# Menghapus kolom yang tidak diperlukan dalan analisa regresi
# Menghapus kolom ID, karena kolom ini hanya berisikan data identiitas
df_gabung.drop('ID', axis = 1, inplace = True)
df_gabung.drop('RESUME_STATUS', axis = 1, inplace = True)

# Memisahkan dataset menjadi fitur (X) dan target (y)
# target
y = df_gabung['CUSTOMER_CATEGORY']
# feature
X = df_gabung.drop('CUSTOMER_CATEGORY', axis = 1).copy()

# Memisahkan dataset menjadi data train dan data test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Encoding Data Categorical (CODE_GENDER) to Data Numerical
# Define Function Encoding
def Gender_Category(x):
    if x == 'M':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (CODE_GENDER) to Data Numerical
# Encoding Process
x_train['CODE_GENDER'] = x_train['CODE_GENDER'].apply(Gender_Category)

# Encoding Data Test
# Encoding Data Categorical (CODE_GENDER) to Data Numerical
# Encoding Process
x_test['CODE_GENDER'] = x_test['CODE_GENDER'].apply(Gender_Category)

# Encoding Data Categorical (FLAG_OWN_CAR) to Data Numerical
# Define Function Encoding
def Car_Category(x):
    if x == 'Y':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (FLAG_OWN_CAR) to Data Numerical
# Encoding Process
x_train['FLAG_OWN_CAR'] = x_train['FLAG_OWN_CAR'].apply(Car_Category)

# Encoding Data Test
# Encoding Data Categorical (FLAG_OWN_CAR) to Data Numerical
# Encoding Process
x_test['FLAG_OWN_CAR'] = x_test['FLAG_OWN_CAR'].apply(Car_Category)

# Encoding Data Categorical (FLAG_OWN_REALTY) to Data Numerical
# Define Function Encoding
def Realty_Category(x):
    if x == 'Y':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (FLAG_OWN_REALTY) to Data Numerical
# Encoding Process
x_train['FLAG_OWN_REALTY'] = x_train['FLAG_OWN_REALTY'].apply(Realty_Category)

# Encoding Data Test
# Encoding Data Categorical (FLAG_OWN_REALTY) to Data Numerical
# Encoding Process
x_test['FLAG_OWN_REALTY'] = x_test['FLAG_OWN_REALTY'].apply(Realty_Category)

# Encoding Data Categorical (NAME_INCOME_TYPE) to Data Numerical
# Define Function Encoding
def Income_Category(x):
    if x == 'Working':
        return 4
    elif x == 'Commercial associate':
        return 3
    elif x == 'State servant':
        return 2
    elif x == 'Pensioner':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (NAME_INCOME_TYPE) to Data Numerical
# Encoding Process
x_train['NAME_INCOME_TYPE'] = x_train['NAME_INCOME_TYPE'].apply(Income_Category)

# Encoding Data Test
# Encoding Data Categorical (NAME_INCOME_TYPE) to Data Numerical
# Encoding Process
x_test['NAME_INCOME_TYPE'] = x_test['NAME_INCOME_TYPE'].apply(Income_Category)

# Encoding Data Categorical (NAME_EDUCATION_TYPE) to Data Numerical
# Define Function Encoding
def Education_Category(x):
    if x == 'Secondary / secondary special':
        return 4
    elif x == 'Higher education':
        return 3
    elif x == 'Incomplete higher':
        return 2
    elif x == 'Lower secondary':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (NAME_EDUCATION_TYPE) to Data Numerical
# Encoding Process
x_train['NAME_EDUCATION_TYPE'] = x_train['NAME_EDUCATION_TYPE'].apply(Education_Category)

# Encoding Data Test
# Encoding Data Categorical (NAME_EDUCATION_TYPE) to Data Numerical
# Encoding Process
x_test['NAME_EDUCATION_TYPE'] = x_test['NAME_EDUCATION_TYPE'].apply(Education_Category)

# Encoding Data Categorical (NAME_FAMILY_STATUS) to Data Numerical
# Define Function Encoding
def Family_Category(x):
    if x == 'Married':
        return 4
    elif x == 'Single / not married':
        return 3
    elif x == 'Civil marriage':
        return 2
    elif x == 'Separated':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (NAME_FAMILY_STATUS) to Data Numerical
# Encoding Process
x_train['NAME_FAMILY_STATUS'] = x_train['NAME_FAMILY_STATUS'].apply(Family_Category)

# Encoding Data Test
# Encoding Data Categorical (NAME_FAMILY_STATUS) to Data Numerical
# Encoding Process
x_test['NAME_FAMILY_STATUS'] = x_test['NAME_FAMILY_STATUS'].apply(Family_Category)

# Encoding Data Categorical (NAME_HOUSING_TYPE) to Data Numerical
# Define Function Encoding
def House_Category(x):
    if x == 'House / apartment':
        return 5
    elif x == 'With parents':
        return 4
    elif x == 'Municipal apartment':
        return 3
    elif x == 'Rented apartment':
        return 2
    elif x == 'Office apartment':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (NAME_HOUSING_TYPE) to Data Numerical
# Encoding Process
x_train['NAME_HOUSING_TYPE'] = x_train['NAME_HOUSING_TYPE'].apply(House_Category)

# Encoding Data Test
# Encoding Data Categorical (NAME_HOUSING_TYPE) to Data Numerical
# Encoding Process
x_test['NAME_HOUSING_TYPE'] = x_test['NAME_HOUSING_TYPE'].apply(House_Category)

# Encoding Data Categorical (OCCUPATION_TYPE) to Data Numerical
# Define Function Encoding
def Occupation_Category(x):
    if x == 'Laborers':
        return 17
    elif x == 'Core staff':
        return 16
    elif x == 'Sales staff':
        return 15
    elif x == 'Managers':
        return 14
    elif x == 'Drivers':
        return 13
    elif x == 'High skill tech staff':
        return 12
    elif x == 'Accountants':
        return 11
    elif x == 'Medicine staff':
        return 10
    elif x == 'Cooking staff':
        return 9
    elif x == 'Security staff':
        return 8
    elif x == 'Cleaning staff':
        return 7
    elif x == 'Private service staff':
        return 6
    elif x == 'Low-skill Laborers':
        return 5
    elif x == 'Waiters/barmen staff':
        return 4
    elif x == 'Secretaries':
        return 3
    elif x == 'HR staff':
        return 2
    elif x == 'Realty agents':
        return 1
    else:
        return 0
    
# Encoding Data Train
# Encoding Data Categorical (OCCUPATION_TYPE) to Data Numerical
# Encoding Proess
x_train['OCCUPATION_TYPE'] = x_train['OCCUPATION_TYPE'].apply(Occupation_Category)

# Encoding Data Test
# Encoding Data Categorical (OCCUPATION_TYPE) to Data Numerical
# Encoding Proess
x_test['OCCUPATION_TYPE'] = x_test['OCCUPATION_TYPE'].apply(Occupation_Category)

# Scaling Data Train and Data Test
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.select_dtypes(['int','float']))
x_test_scaled = scaler.transform(x_test.select_dtypes(['int','float']))
x_train_scaled = pd.DataFrame(
    x_train_scaled,
    columns=x_train.select_dtypes(['int', 'float']).columns,
    index=x_train.index
)
x_test_scaled = pd.DataFrame(
    x_test_scaled,
    columns=x_test.select_dtypes(['int', 'float']).columns,
    index=x_test.index
)

# Troubleshoot Class Imbalance dengan Teknik Resampling SMOTE pada Data Train
# SMOTE : Menambahkan data sintetis untuk kelas minoritas
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)

# Define Final Dataframe for Modelling
x_train_final = x_train_resampled
y_train_final = y_train_resampled
x_test_final = x_test_scaled
y_test_final = y_test

# Develop FUNCTION to display Model Evaluation Metrics
# Accuracy = mengukur kesesuaian prediksi model, semakin tinggi, semakin baik
# Recall = mengukur kesesuaian prediksi positif, semakin tinggi, semakin baik
# Precision = mengukur kesesuaian prediksi positif, semakin tinggi, semakin baik
# F1-Score = harmonik rata-rata dari Precision dan Recall, mengukur kesesuaian model secara keseluruhan

def evaluate_model(true, prediction, label = None):

  temp_result = pd.DataFrame({'Remark': label,
                              'Accuracy': [accuracy_score(true, prediction)*100],
                              'Recall' : [recall_score(true, prediction)*100],
                              'Precision' : precision_score(true, prediction)*100,
                              'F1-Score' : f1_score(true, prediction)*100,
                              'ROC-AUC' : roc_auc_score(true, prediction)*100,
                              })
  return temp_result


def display_confusion_matrix(true, prediction):

  cm = confusion_matrix(true, prediction)

  # Create a DataFrame for better display
  cm_df = pd.DataFrame(
      cm,
      index= ['Actual: Not Approved', 'Actual: Approved'],
      columns=['Predicted: Not Approved', 'Predicted: Approved']
  )

  cm_df.loc['Total'] = cm_df.sum(axis=0)
  cm_df['Total'] = cm_df.sum(axis=1)

  return cm_df

# Single select menu

options = ["No Select", "Logistic Regression", "KNN Regression", "Decision Tree Regression", "Random Forest Regression", "XGBoost Regression"]

selected = st.selectbox("Please Select Regression Method :", options)

if selected == "Logistic Regression":
    st.write("Basic Model :")
    st.write("Parameter Set : max_iter=5000, n_jobs=-1, random_state=42")
    model_logistic = LogisticRegression(max_iter=5000, n_jobs=-1, random_state=42)
    model_logistic.fit(x_train_final, y_train_final)
    pred_train_logistic_before = model_logistic.predict(x_train_final)
    pred_logistic_before = model_logistic.predict(x_test_final)
    dflog_before = pd.concat([evaluate_model(y_train_final, pred_train_logistic_before, "Train"),
                              evaluate_model(y_test_final, pred_logistic_before, "Test")])
    st.dataframe(dflog_before)
    dflogcm_before = display_confusion_matrix(y_test_final, pred_logistic_before)
    st.dataframe(dflogcm_before)
    st.write("Hyper Parameter Tuning :")
    st.write("Parameter Set : max_iter=5000, n_jobs=-1, random_state=42, C=6.366859160799429, penalty=l2, class_weight=balanced")
    model_logistic_after = LogisticRegression(max_iter=1000, random_state=42, C=6.366859160799429, penalty='l2', class_weight='balanced')
    model_logistic_after.fit(x_train_final, y_train_final)
    pred_train_logistic_after = model_logistic_after.predict(x_train_final)
    pred_logistic_after = model_logistic_after.predict(x_test_final)
    dflog_after = pd.concat([evaluate_model(y_train_final, pred_train_logistic_after, "Train"),
                             evaluate_model(y_test_final, pred_logistic_after, "Test")])
    st.dataframe(dflog_after)
    dflogcm_after = display_confusion_matrix(y_test_final, pred_logistic_after)
    st.dataframe(dflogcm_after)

elif selected == "KNN Regression":
    st.write("Basic Model :")
    st.write("Parameter Set : None")
    model_knn = KNeighborsClassifier()
    model_knn.fit(x_train_final, y_train_final)
    pred_train_knn_before = model_knn.predict(x_train_final)
    pred_knn_before = model_knn.predict(x_test_final)
    dfknn_before = pd.concat([evaluate_model(y_train_final, pred_train_knn_before, "Train"),
                              evaluate_model(y_test_final, pred_knn_before, "Test")])
    st.dataframe(dfknn_before)
    dfknncm_before = display_confusion_matrix(y_test_final, pred_knn_before)
    st.dataframe(dfknncm_before)
    st.write("Hyper Parameter Tuning :")
    st.write("Parameter Set : leaf_size=40, metric=minkowski, n_neighbors=4, p=2, weights=distance")
    model_knn_after = KNeighborsClassifier(leaf_size=40, metric='minkowski', n_neighbors=4, p=2, weights='distance')
    model_knn_after.fit(x_train_final, y_train_final)
    pred_train_knn_after = model_knn_after.predict(x_train_final)
    pred_knn_after = model_knn_after.predict(x_test_final)
    dfknn_after = pd.concat([evaluate_model(y_train_final, pred_train_knn_after, "Train"),
                             evaluate_model(y_test_final, pred_knn_after, "Test")])
    st.dataframe(dfknn_after)
    dfknncm_after = display_confusion_matrix(y_test_final, pred_knn_after)
    st.dataframe(dfknncm_after)

elif selected == "Decision Tree Regression":
    st.write("Basic Model :")
    st.write("Parameter Set : random_state=42")
    model_tree = DecisionTreeClassifier(random_state=42)
    model_tree.fit(x_train_final, y_train_final)
    pred_train_tree_before = model_tree.predict(x_train_final)
    pred_tree_before = model_tree.predict(x_test_final)
    dftree_before = pd.concat([evaluate_model(y_train_final, pred_train_tree_before, "Train"),
                       evaluate_model(y_test_final, pred_tree_before, "Test")])
    st.dataframe(dftree_before)
    dftreecm_before = display_confusion_matrix(y_test_final, pred_tree_before)
    st.dataframe(dftreecm_before)
    st.write("Hyper Parameter Tuning :")
    st.write("Parameter Set : random_state=42, ccp_alpha=0.001, criterion=log_loss, max_depth=28, max_features=sqrt, min_samples_leaf=2, min_samples_split=27, splitter=best")
    model_tree_after = DecisionTreeClassifier(random_state=42, ccp_alpha=0.001, criterion='log_loss', max_depth=28, max_features='sqrt', min_samples_leaf=2, min_samples_split=27, splitter='best')
    model_tree_after.fit(x_train_final, y_train_final)
    pred_train_tree_after = model_tree_after.predict(x_train_final)
    pred_tree_after = model_tree_after.predict(x_test_final)
    dftree_after = pd.concat([evaluate_model(y_train_final, pred_train_tree_after, "Train"),
                              evaluate_model(y_test_final, pred_tree_after, "Test")])
    st.dataframe(dftree_after)
    dftreecm_after = display_confusion_matrix(y_test_final, pred_tree_after)
    st.dataframe(dftreecm_after)

elif selected == "Random Forest Regression":
    st.write("Basic Model :")
    st.write("Parameter Set : random_state=42, n_jobs=-1")
    model_rfo = RandomForestClassifier(random_state=42, n_jobs=-1)
    model_rfo.fit(x_train_final, y_train_final)
    pred_train_rfo_before = model_rfo.predict(x_train_final)
    pred_rfo_before = model_rfo.predict(x_test_final)
    dfrfo_before = pd.concat([evaluate_model(y_train_final, pred_train_rfo_before, "Train"),
                              evaluate_model(y_test_final, pred_rfo_before, "Test")])
    st.dataframe(dfrfo_before)
    dfrfocm_before = display_confusion_matrix(y_test_final, pred_rfo_before)
    st.dataframe(dfrfocm_before)
    st.write("Hyper Parameter Tuning :")
    st.write("Parameter Set : n_estimators=192, max_depth=20, random_state=42, n_jobs=-1, max_features=log2, max_samples=0.85, min_samples_leaf=1, min_samples_split=6")
    model_rfo_after = RandomForestClassifier(n_estimators=192, max_depth=20, random_state=42, n_jobs=-1, max_features='log2', max_samples=0.85, min_samples_leaf=1, min_samples_split=6)
    model_rfo_after.fit(x_train_final, y_train_final)
    pred_train_rfo_after = model_rfo_after.predict(x_train_final)
    pred_rfo_after = model_rfo_after.predict(x_test_final)
    dfrfo_after = pd.concat([evaluate_model(y_train_final, pred_train_rfo_after, "Train"),
                             evaluate_model(y_test_final, pred_rfo_after, "Test")])
    st.dataframe(dfrfo_after)
    dfrfocm_after = display_confusion_matrix(y_test_final, pred_rfo_after)           
    st.dataframe(dfrfocm_after)

elif selected == "XGBoost Regression":
    st.write("ok5")
else:
    st.write("Please Select an Option First")


#st.write("Maximum Range of Date     : 2014-01-03 to 2017-12-30")
#col_c1, col_c2 = st.columns(2)
#with col_c1:
#    start_date = st.date_input("Input Start Date of Data Analysis", datetime.date(2014, 1, 3))
#with col_c2:
#    end_date = st.date_input("Input End Date of Data Analysis", datetime.date(2017, 12, 30))

#df pd.read_csv('Sample_Superstore.csv', encoding = "ISO-8859-1")
#df['Order Date'] = pd.to_datetime(df['Order Date']).dt.date

#date_mask = (df['Order Date'] > start_date) & (df['Order Date'] <= end_date)
#df_date = df.loc[date_mask].sort_values(by = 'Order Date',ascending=True)

#total_tsc = df_date['Order ID'].count()
#total_rvn = df_date['Sales'].sum().round(1)
#total_pft = df_date['Profit'].sum().round(1)

#col_d1, col_d2, col_d3 = st.columns(3, border=True, vertical_alignment="center")
#with col_d1:
#    st.write("Total Transaction")
#    st.header(total_tsc)
#with col_d2:
#    st.write("Total Revenue")
#    st.header(total_rvn)
#with col_d3:
#    st.write("Total Profit")
#    st.header(total_pft)
#st.write("")

#df_tps = df_date.groupby('State')['Order ID'].count().sort_values(ascending=False).reset_index()
#df_tps.rename(columns={'Order ID': 'Total Transaction'}, inplace=True)
#df_rps = df_date.groupby('State')['Sales'].sum().sort_values(ascending=False).round(1).reset_index()
#df_rps.rename(columns={'Sales': 'Total Revenue'}, inplace=True)
#df_pps = df_date.groupby('State')['Profit'].sum().sort_values(ascending=False).round(1).reset_index()
#df_pps.rename(columns={'Profit': 'Total Profit'}, inplace=True)

#st.write("Total Transaction per State")
#st.bar_chart(df_tps, x='State', y='Total Transaction', color="#fd0")
#st.write("Total Revenue per State")
#st.bar_chart(df_rps, x='State', y='Total Revenue')
#st.write("Total Profit per State")
#st.bar_chart(df_pps, x='State', y='Total Profit',color="#f0f")

#st.write("Resume Transaction Data per State")
#df_state_1 = pd.merge(df_tps,df_rps, on = 'State', how = 'inner')
#df_state_2 = pd.merge(df_state_1,df_pps, on = 'State', how = 'inner')
#st.dataframe(df_state_2)

#st.write("Detail Transaction Data")
#st.dataframe(df_date)

