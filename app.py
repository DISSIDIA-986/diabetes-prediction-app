"""
Diabetes Prediction Application
End-to-end ML application using PyCaret model deployed on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.classification import load_model, predict_model

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Data source
DATA_URL = 'https://dissidia.oss-cn-beijing.aliyuncs.com/IntegratedAI/PredictiveAnalytics/Dataset/diabetes.csv'

@st.cache_data
def load_data():
    """Load the diabetes dataset"""
    return pd.read_csv(DATA_URL)

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

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA", "ℹ️ About"])

    # Load data for EDA
    data = load_data()

    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        st.header("Make a Prediction")

        col_input, col_result = st.columns([1, 1])

        with col_input:
            st.subheader("Patient Information")

            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.slider(
                    "Pregnancies", min_value=0, max_value=17, value=3,
                    help="Number of times pregnant"
                )
                glucose = st.slider(
                    "Glucose (mg/dL)", min_value=0, max_value=200, value=120,
                    help="Plasma glucose concentration"
                )
                blood_pressure = st.slider(
                    "Blood Pressure (mm Hg)", min_value=0, max_value=130, value=70,
                    help="Diastolic blood pressure"
                )
                skin_thickness = st.slider(
                    "Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                    help="Triceps skin fold thickness"
                )

            with col2:
                insulin = st.slider(
                    "Insulin (mu U/ml)", min_value=0, max_value=850, value=80,
                    help="2-Hour serum insulin"
                )
                bmi = st.slider(
                    "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.0, step=0.1,
                    help="Body mass index"
                )
                dpf = st.slider(
                    "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47, step=0.01,
                    help="Diabetes likelihood based on family history"
                )
                age = st.slider(
                    "Age (years)", min_value=21, max_value=81, value=33
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

        with col_result:
            st.subheader("Prediction Result")

            try:
                model = load_diabetes_model()
                prediction = predict_model(model, data=input_data)

                pred_label = prediction['prediction_label'].iloc[0]
                pred_score = prediction['prediction_score'].iloc[0]

                if pred_label == 1:
                    st.error("⚠️ **High Risk of Diabetes**")
                    st.metric("Confidence", f"{pred_score:.1%}")
                    st.markdown("""
                    **Recommendations:**
                    - Consult with a healthcare provider
                    - Consider lifestyle modifications
                    - Regular glucose monitoring
                    """)
                else:
                    st.success("✅ **Low Risk of Diabetes**")
                    st.metric("Confidence", f"{pred_score:.1%}")
                    st.markdown("""
                    **Recommendations:**
                    - Maintain healthy lifestyle
                    - Regular health checkups
                    - Balanced diet and exercise
                    """)

                # Show input summary
                st.markdown("---")
                st.markdown("**Your Input:**")
                st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

    # ==================== TAB 2: EDA ====================
    with tab2:
        st.header("Exploratory Data Analysis")

        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(data))
        col2.metric("Features", len(data.columns) - 1)
        col3.metric("Diabetic Cases", f"{data['Outcome'].sum()} ({data['Outcome'].mean()*100:.1f}%)")
        col4.metric("Non-Diabetic", f"{len(data) - data['Outcome'].sum()} ({(1-data['Outcome'].mean())*100:.1f}%)")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(data, use_container_width=True)

        st.divider()

        # Distribution plots
        st.subheader("📈 Feature Distributions")

        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        selected_feature = st.selectbox("Select feature to explore:", features)

        col1, col2 = st.columns(2)

        with col1:
            # Histogram with outcome overlay
            fig = px.histogram(
                data, x=selected_feature, color='Outcome',
                barmode='overlay', opacity=0.7,
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'Outcome': 'Diabetes'},
                title=f'Distribution of {selected_feature} by Outcome'
            )
            fig.update_layout(legend_title_text='Diabetes')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot
            fig = px.box(
                data, x='Outcome', y=selected_feature,
                color='Outcome',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'Outcome': 'Diabetes'},
                title=f'{selected_feature} by Diabetes Status'
            )
            fig.update_xaxes(tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes'])
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")

        corr_matrix = data.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Feature Correlation Matrix'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Pairwise scatter plots
        st.subheader("🔍 Feature Relationships")

        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis:", features, index=1)  # Default: Glucose
        with col2:
            y_feature = st.selectbox("Y-axis:", features, index=5)  # Default: BMI

        fig = px.scatter(
            data, x=x_feature, y=y_feature, color='Outcome',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            opacity=0.6,
            title=f'{x_feature} vs {y_feature}',
            labels={'Outcome': 'Diabetes'}
        )
        fig.update_layout(legend_title_text='Diabetes')
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Statistics table
        st.subheader("📊 Descriptive Statistics")

        stats_by_outcome = data.groupby('Outcome').describe().T

        tab_all, tab_no_diabetes, tab_diabetes = st.tabs(["All Data", "No Diabetes (0)", "Diabetes (1)"])

        with tab_all:
            st.dataframe(data.describe().T.round(2), use_container_width=True)
        with tab_no_diabetes:
            st.dataframe(data[data['Outcome'] == 0].describe().T.round(2), use_container_width=True)
        with tab_diabetes:
            st.dataframe(data[data['Outcome'] == 1].describe().T.round(2), use_container_width=True)

    # ==================== TAB 3: ABOUT ====================
    with tab3:
        st.header("About This Application")

        st.markdown("""
        ### Dataset
        **Pima Indians Diabetes Dataset** from the National Institute of Diabetes and Digestive and Kidney Diseases.

        - **Samples**: 768 female patients
        - **Features**: 8 diagnostic measurements
        - **Target**: Binary classification (0 = No Diabetes, 1 = Diabetes)

        ### Features Description
        | Feature | Description |
        |---------|-------------|
        | Pregnancies | Number of times pregnant |
        | Glucose | Plasma glucose concentration (2h oral glucose tolerance test) |
        | BloodPressure | Diastolic blood pressure (mm Hg) |
        | SkinThickness | Triceps skin fold thickness (mm) |
        | Insulin | 2-Hour serum insulin (mu U/ml) |
        | BMI | Body mass index (weight in kg/(height in m)²) |
        | DiabetesPedigreeFunction | Diabetes likelihood based on family history |
        | Age | Age in years |

        ### Model Architecture
        **Stacked Ensemble** combining multiple algorithms:
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        - AdaBoost
        - K-Nearest Neighbors

        With **Logistic Regression** as the meta-learner.

        ### Preprocessing Pipeline
        - Z-score normalization
        - Feature selection
        - Multicollinearity removal
        - SMOTE for class imbalance handling

        ### Built With
        - **Streamlit** - Web application framework
        - **PyCaret** - Automated machine learning
        - **Plotly** - Interactive visualizations
        """)

        st.divider()
        st.caption("Predictive Analytics Course Project | Built with Streamlit and PyCaret")

if __name__ == "__main__":
    main()
