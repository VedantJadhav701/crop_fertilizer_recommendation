import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tracemalloc

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")
st.title("🌿 Smart Farming Assistant: Crop & Fertilizer Recommendation")

# Load datasets
@st.cache_data
def load_fertilizer_data():
    return pd.read_csv("Fertilizer.csv")

@st.cache_data
def load_crop_data():
    return pd.read_csv("Crop_recommendation.csv")

# Tabs
tab1, tab2 = st.tabs(["Fertilizer Recommendation", "Crop Recommendation"])

# ------------------------------ FERTILIZER TAB ------------------------------ 
# ------------------------------ FERTILIZER TAB ------------------------------ 
with tab1:
    df = load_fertilizer_data()
    st.subheader("Fertilizer Dataset Preview")
    st.dataframe(df.head())

    if st.checkbox("Show Fertilizer Data Visualizations"):
        st.subheader("Correlation Heatmap")
        fig3 = plt.figure(figsize=(8, 5))
        sns.heatmap(df.drop('Fertilizer Name', axis=1).corr(), annot=True, cmap='coolwarm')
        st.pyplot(fig3)

    label_encoder = LabelEncoder()
    df['Fertilizer Name'] = label_encoder.fit_transform(df['Fertilizer Name'])

    X = df[['Nitrogen', 'Potassium', 'Phosphorous']]
    y = df['Fertilizer Name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def train_fert_model(model):
        tracemalloc.start()
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        duration = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        y_pred = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'log_loss': log_loss(y_test, probs),
            'conf_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, target_names=label_encoder.classes_),
            'model': model,
            'time': duration,
            'memory': current / 1e6,
            'peak_memory': peak / 1e6
        }

    model_type = st.sidebar.selectbox("Select Fertilizer Model", ["Random Forest"])
    
    # Initialize the model variable in session state if it's not already there
    if 'fert_model' not in st.session_state:
        st.session_state.fert_model = None

    if st.button("Train Fertilizer Model"):
        results = train_fert_model(RandomForestClassifier(random_state=42))
        st.session_state.fert_model = results['model']  # Save the trained model in session state

        st.success("Model Trained ✅")
        st.code(results['report'])
        st.json({
            "Accuracy": results['accuracy'],
            "Precision": results['precision'],
            "Recall": results['recall'],
            "F1 Score": results['f1_score'],
            "Training Time (s)": results['time'],
            "Memory (MB)": results['memory'],
            "Peak Memory (MB)": results['peak_memory'],
            "Log Loss": results['log_loss']
        })

        fig4, ax = plt.subplots()
        sns.heatmap(results['conf_matrix'], annot=True, fmt='g', cmap='Blues', ax=ax)
        st.pyplot(fig4)

    st.sidebar.subheader("🔍 Predict Fertilizer")
    n = st.sidebar.number_input("Nitrogen", 0)
    p = st.sidebar.number_input("Phosphorous", 0)
    k = st.sidebar.number_input("Potassium", 0)

    if st.sidebar.button("Get Fertilizer Recommendation"):
        # Check if the model is trained and stored in session state
        if st.session_state.fert_model is None:
            st.sidebar.error("Please train the model first.")
        else:
            input_scaled = scaler.transform([[n, k, p]])
            prediction = st.session_state.fert_model.predict(input_scaled)
            st.sidebar.success(f"Recommended Fertilizer: {label_encoder.inverse_transform(prediction)[0]}")



# ------------------------------ CROP TAB ------------------------------
with tab2:
    crop_df = load_crop_data()
    st.subheader("Crop Recommendation Dataset Preview")
    st.dataframe(crop_df.head())

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'

    X_crop = crop_df[features]
    y_crop = crop_df[target]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
    scaler_crop = StandardScaler()
    Xc_train_scaled = scaler_crop.fit_transform(Xc_train)
    Xc_test_scaled = scaler_crop.transform(Xc_test)

    crop_model = RandomForestClassifier(random_state=42)
    crop_model.fit(Xc_train_scaled, yc_train)

    st.sidebar.subheader("🌾 Predict Suitable Crop")
    crop_input = [
        st.sidebar.number_input("Nitrogen (N)", 0),
        st.sidebar.number_input("Phosphorous (P)", 0),
        st.sidebar.number_input("Potassium (K)", 0),
        st.sidebar.number_input("Temperature (°C)", 0.0),
        st.sidebar.number_input("Humidity (%)", 0.0),
        st.sidebar.number_input("pH", 0.0),
        st.sidebar.number_input("Rainfall (mm)", 0.0)
    ]

    if st.sidebar.button("Get Crop Recommendation"):
        input_crop_scaled = scaler_crop.transform([crop_input])
        crop_prediction = crop_model.predict(input_crop_scaled)[0]
        st.sidebar.success(f"Recommended Crop: {crop_prediction}")
