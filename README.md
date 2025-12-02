German Credit Risk Assessment

An end-to-end machine learning project that predicts whether a loan applicant is a good or bad credit risk using the German Credit Dataset. This project covers the complete workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and deployment using a Streamlit web application.

📌 Project Overview

This project aims to build a reliable Credit Risk Prediction System capable of assisting financial institutions in assessing loan applicants. Using the ExtraTreesClassifier model, the system analyzes features such as age, housing status, savings/checking accounts, credit amount, and purpose of the loan.

The final model is deployed through an interactive Streamlit app that provides real-time predictions.

📂 Project Structure
├── german_credit_data.csv
├── German Credit Risk Assessment.ipynb
├── app.py
├── ExtraTreesClassifier_GermanCreditModel.pkl
├── <encoders>.pkl
└── README.md

🔍 Exploratory Data Analysis (EDA)

- Distribution of key features (Age, Purpose, Housing, Credit Amount)

- Countplots for categorical features

- Boxplots for outlier detection

- Correlation heatmap for numerical features

- Risk class imbalance inspection

- These steps help understand patterns and identify risk indicators.

🛠 Data Preprocessing

- Handling missing values

- Encoding categorical features using Label Encoders

- Storing encoders with Joblib

- Splitting dataset into train-test sets

- Scaling and cleaning numerical features

- Encoders are reused inside the Streamlit app to ensure consistent predictions.

🤖 Model Development

The following models were tested:

- Decision Tree

- Random Forest

- Extra Trees

ExtraTreesClassifier showed the best results due to:

- High accuracy

- Strong handling of mixed feature types

- Robustness to noise and overfitting

The final model is exported as:

ExtraTreesClassifier_GermanCreditModel.pkl

🚀 Streamlit App

The Streamlit interface allows users to input:

- Age

- Credit amount

- Duration

- Employment

- Gender

- Housing

- Purpose

- Saving accounts

- checking accounts

The app loads the saved model and encoders to generate real-time credit risk predictions.

📊 Evaluation

Model performance measured using:

- Accuracy

- Classification report

- Confusion matrix

- Results show strong performance in predicting good credit risk, with reasonable detection of bad credit risk.

📘 Conclusion

This project demonstrates a full end-to-end ML workflow, from EDA to deployment. It is a strong showcase of skills in:

- Data preprocessing

- Feature engineering

- ML model development

- Model evaluation

- Deployment with Streamlit

The system is production-ready and can be extended with more advanced preprocessing, hyperparameter tuning, or risk scoring strategies.

📎 Technologies Used

- Python

- Pandas

- NumPy

- Matplotlib / Seaborn

- Scikit-learn

- Joblib

- Streamlit
