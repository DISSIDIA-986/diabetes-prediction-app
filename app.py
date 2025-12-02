"""
Diabetes Prediction Application
End-to-end ML application using PyCaret model deployed on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_diabetes_model():
    """Load the pre-trained PyCaret model"""
    model = load_model('diabetes_model')
    return model

# Main app
def main():
    st.title("🩺 Diabetes Prediction Application")
    st.markdown("""
    This application predicts the likelihood of diabetes based on diagnostic measurements.
    The model was trained using the Pima Indians Diabetes Dataset with PyCaret's automated ML pipeline.
    """)

    st.divider()

    # Sidebar for input
    st.sidebar.header("Patient Information")
    st.sidebar.markdown("Enter the patient's diagnostic measurements:")

    # Input features
    pregnancies = st.sidebar.slider(
        "Pregnancies",
        min_value=0, max_value=17, value=3,
        help="Number of times pregnant"
    )

    glucose = st.sidebar.slider(
        "Glucose (mg/dL)",
        min_value=0, max_value=200, value=120,
        help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)"
    )

    blood_pressure = st.sidebar.slider(
        "Blood Pressure (mm Hg)",
        min_value=0, max_value=130, value=70,
        help="Diastolic blood pressure"
    )

    skin_thickness = st.sidebar.slider(
        "Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )

    insulin = st.sidebar.slider(
        "Insulin (mu U/ml)",
        min_value=0, max_value=850, value=80,
        help="2-Hour serum insulin"
    )

    bmi = st.sidebar.slider(
        "BMI (kg/m²)",
        min_value=0.0, max_value=70.0, value=32.0, step=0.1,
        help="Body mass index (weight in kg/(height in m)²)"
    )

    dpf = st.sidebar.slider(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=2.5, value=0.47, step=0.01,
        help="A function which scores likelihood of diabetes based on family history"
    )

    age = st.sidebar.slider(
        "Age (years)",
        min_value=21, max_value=81, value=33,
        help="Age in years"
    )

    # Create input dataframe
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Display input summary
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Patient Data Summary")
        st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

    with col2:
        st.subheader("🔮 Prediction")

        # Make prediction
        try:
            model = load_diabetes_model()
            prediction = predict_model(model, data=input_data)

            pred_label = prediction['prediction_label'].iloc[0]
            pred_score = prediction['prediction_score'].iloc[0]

            if pred_label == 1:
                st.error(f"⚠️ **High Risk of Diabetes**")
                st.metric("Confidence", f"{pred_score:.1%}")
                st.markdown("""
                **Recommendations:**
                - Consult with a healthcare provider
                - Consider lifestyle modifications
                - Regular glucose monitoring
                """)
            else:
                st.success(f"✅ **Low Risk of Diabetes**")
                st.metric("Confidence", f"{pred_score:.1%}")
                st.markdown("""
                **Recommendations:**
                - Maintain healthy lifestyle
                - Regular health checkups
                - Balanced diet and exercise
                """)

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please ensure the model file 'diabetes_model.pkl' is in the app directory.")

    # Model information
    st.divider()
    st.subheader("ℹ️ About the Model")
    st.markdown("""
    **Dataset**: Pima Indians Diabetes Dataset (768 samples, 8 features)

    **Features Used**:
    - Pregnancies, Glucose, Blood Pressure, Skin Thickness
    - Insulin, BMI, Diabetes Pedigree Function, Age

    **Model**: Stacked Ensemble (Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, KNN)

    **Preprocessing**: Z-score normalization, feature selection, class imbalance handling
    """)

    # Footer
    st.divider()
    st.caption("Built with Streamlit and PyCaret | Predictive Analytics Course Project")

if __name__ == "__main__":
    main()
