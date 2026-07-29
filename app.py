import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt

# 1. Configuración de la página
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Configuración de MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "Churn_Production_Model"
MODEL_ALIAS = "Champion"


# 2. Carga optimizada del modelo
@st.cache_resource
def load_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)

try:
    model = load_champion_model()
except Exception as e:
    st.error(f"Error al cargar el modelo desde MLflow: {e}")
    st.stop()

# 3. Interfaz de usuario
st.title("🔮 Predicción de Churn de Clientes")
st.markdown("Esta aplicación utiliza el modelo **@Champion** registrado en **MLflow** para evaluar el riesgo de cancelación en tiempo real y explicar las causas con **SHAP**.")

st.divider()

# Formulario dividido en 3 columnas
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Perfil del Cliente")
    gender = st.selectbox("Género", ["Female", "Male"])
    senior = st.selectbox("Adulto Mayor (Senior)", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    partner = st.selectbox("Tiene Pareja", ["Yes", "No"])
    dependents = st.selectbox("Tiene Dependientes", ["Yes", "No"])
    tenure = st.slider("Antigüedad (Meses de tenure)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("📡 Servicios Contratados")
    phone_service = st.selectbox("Servicio Telefónico", ["Yes", "No"])
    multiple_lines = st.selectbox("Líneas Múltiples", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Backups Online", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Protección de Dispositivos", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Películas", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Contrato y Pagos")
    contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Facturación Electrónica", ["Yes", "No"])
    payment_method = st.selectbox("Método de Pago", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Cargo Mensual ($)", min_value=18.0, max_value=120.0, value=65.0)
    total_charges = st.number_input("Cargos Totales ($)", min_value=0.0, max_value=9000.0, value=float(tenure * monthly_charges))

st.divider()

# 4. Botón de Inferencia
if st.button("🚀 Evaluar Riesgo de Churn", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Convertir nombres de columnas a minúsculas
    input_data.columns = input_data.columns.str.lower()

    # Predicción
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Mostrar Resultados de la Inferencia
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Probabilidad de Churn", value=f"{probability * 100:.1f}%")

    with res_col2:
        if probability >= 0.65:
            st.error("🚨 **NIVEL DE RIESGO: ALTO**\n\nEste cliente tiene una alta probabilidad de cancelar su servicio.")
        elif probability >= 0.35:
            st.warning("⚠️ **NIVEL DE RIESGO: MEDIO**\n\nEl cliente muestra señales moderadas de insatisfacción.")
        else:
            st.success("✅ **NIVEL DE RIESGO: BAJO**\n\nEl cliente se encuentra fidelizado.")

    # 5. Explicabilidad Local con SHAP
    st.divider()
    st.subheader("🔍 Explicación Explicable (SHAP) del Cliente Seleccionado")
    st.caption("Los bloques **rojos** empujan la probabilidad de Churn hacia arriba (aumentan riesgo), mientras que los **azules** la reducen.")

    try:
        # Extraer los pasos del Pipeline
        preprocessor = model.named_steps['preprocessor']
        clf = model.named_steps['clf']

        # Preprocesar la entrada del cliente
        X_transformed = preprocessor.transform(input_data)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Nombres de variables limpios
        feature_names = [f.split('__')[-1] for f in preprocessor.get_feature_names_out()]

        # Crear explainer de SHAP
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(X_transformed)

        # Si es clasificación multiclase / binaria con 2 salidas
        if len(shap_values.shape) == 3:
            shap_explanation = shap_values[0, :, 1]
        else:
            shap_explanation = shap_values[0]

        # Asignar nombres de variables al objeto de SHAP
        shap_explanation.feature_names = feature_names

        # Gráfico Waterfall
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_explanation, max_display=7, show=False)
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"No se pudo generar la explicación SHAP: {e}")