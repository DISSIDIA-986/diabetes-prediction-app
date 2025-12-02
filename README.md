# Diabetes Prediction App

End-to-end Machine Learning application for diabetes prediction using PyCaret, deployed on Streamlit Cloud.

## Live Demo

[Open App on Streamlit Cloud](https://diabetes-prediction-app.streamlit.app)

## Features

- Interactive web interface for diabetes risk prediction
- Trained stacked ensemble model (Logistic Regression + Random Forest + Gradient Boosting + AdaBoost + KNN)
- Real-time predictions with confidence scores

## Dataset

Pima Indians Diabetes Dataset (768 samples, 8 features):
- Pregnancies, Glucose, Blood Pressure, Skin Thickness
- Insulin, BMI, Diabetes Pedigree Function, Age

## Model Details

- **Preprocessing**: Z-score normalization, feature selection, SMOTE for class imbalance
- **Architecture**: Stacked ensemble with Logistic Regression as meta-learner
- **Framework**: PyCaret 3.x

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` - Streamlit application
- `diabetes_model.pkl` - Trained PyCaret model
- `requirements.txt` - Python dependencies
