# Telco Customer Churn Predictor & Early Warning System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange)

Un sistema de Machine Learning end-to-end orientado a la **reducción de la tasa de cancelación (Churn)** de clientes en telecomunicaciones. Combina un pipeline modular de datos, rastreo sistemático de experimentos con **MLflow**, selección automática del modelo `@Champion` e interpretabilidad local mediante **SHAP**.

---

## El Problema de Negocio

En la industria de las telecomunicaciones, **adquirir un nuevo cliente cuesta entre 5 y 25 veces más que retener a uno existente**. La cancelación no detectada a tiempo impacta directamente en el *Customer Lifetime Value* (CLV) y en los ingresos recurrentes mensuales (MRR).

### Retos Clave:
1. **Identificación Tardía:** Las empresas suelen enterarse de la insatisfacción cuando el cliente ya solicitó la cancelación.
2. **Costos de Retención Ineficientes:** Ofrecer descuentos masivos a clientes que no pensaban irse genera pérdidas financieras innecesarias.
3. **Desbalance de Clientes:** La gran mayoría de la base de clientes no cancela, lo que dificulta a los modelos tradicionales detectar los verdaderos casos de riesgo.

---

## La Solución de Machine Learning

Esta solución actúa como un **sistema de alerta temprana** para el equipo de *Customer Success* y Retención:

* **Predicción Probabilística:** En lugar de una clasificación binaria rígida, el modelo genera una probabilidad continua de riesgo (0% a 100%).
* **Segmentación Dinámica de Riesgo:**
  * 🟢 **Riesgo Bajo (< 35%):** Cliente fidelizado. Sin costo de acción.
  * 🟡 **Riesgo Medio (35% - 65%):** Campañas automatizadas de *engagement* liviano.
  * 🔴 **Riesgo Alto (> 65%):** Intervención prioritaria con ofertas de retención personalizadas.
* **Explicabilidad Local con SHAP:** Identifica *por qué* cada cliente específico está en riesgo para ofrecer la palanca de retención adecuada (ej. migrar de contrato mensual a anual o añadir soporte técnico).

---

## Arquitectura MLOps End-to-End

El proyecto sigue las mejores prácticas de ingeniería y modularidad para garantizar escalabilidad y mantenibilidad:

│
├── data/                  # Almacenamiento local SQLite y fuentes
├── src/
│   ├── data_pipeline.py   # ETL, limpieza y preprocesamiento modular
│   ├── tracker.py         # Abstracción de registro de métricas y artifacts
│   ├── train_logreg.py    # Baseline interpretable (Logistic Regression)
│   ├── train_rf.py        # Modelo de ensamble bagging (Random Forest)
│   ├── train_xgboost.py   # Gradient Boosting optimizado
│   ├── evaluate.py        # Promoción automática del modelo @Champion al Registry
│
├── app.py                 # Dashboard interactivo con Streamlit y SHAP
├── mlflow.db              # Backend Store para experimentos de MLflow
├── environment.yml        # Entorno reproducible (Conda)
└── README.md

--

## Evaluación de Modelos & Criterio Estratégico

Para evitar la "ilusión" de métricas infladas como ROC-AUC en datos desbalanceados, el pipeline optimiza y selecciona al ganador utilizando **PR-AUC (`average_precision`)**. Esta métrica se enfoca exclusivamente en la precisión de la clase positiva (clientes en riesgo de churn).

| Modelo | Accuracy | PR-AUC (`avg_precision`) | Recall Churn | Estado en Registry |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** 🌲 | **0.810** | **0.670** | 0.505 | 🏆 **`@Champion`** |
| **XGBoost** ⚡ | 0.805 | 0.668 | 0.518 | Evaluado |
| **Logistic Regression** 📈 | 0.804 | 0.666 | **0.521** | Baseline |

### Key Takeaway de Negocio:
El rendimiento cercano entre la Regresión Logística y los modelos basados en árboles demuestra que las relaciones clave de Churn son fuertemente aditivas y directas (antigüedad, tipo de contrato y cargos mensuales), permitiendo inferencias estables sin riesgo severo de *overfitting*.

---

## Interfaz de Usuario & Explicabilidad (Streamlit + SHAP)

La aplicación web integrada conecta directamente con el **MLflow Model Registry** para consumir en tiempo real el modelo marcado con el alias `@Champion`.

* **Entrada de Datos:** Formulario estructurado para simular perfiles de clientes.
* **Inferencia en Tiempo Real:** Evaluación instantánea de probabilidad y nivel de riesgo.
* **XAI (Explainable AI):** Gráfico tipo *Waterfall* de **SHAP** que desglosa visualmente las variables que empujan el riesgo hacia arriba (rojo) o hacia abajo (azul).

---

## Cómo Ejecutar el Proyecto

### 1. Clonar el repositorio y configurar el entorno
```bash
git clone [https://github.com/fpin](https://github.com/fpin)
eda94/Churn.git
cd Churn
conda env create -f environment.yml
conda activate churn_env

### 2. Ejecutar la UI de MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db


### 2. Lanzar la aplicación interactiva de Streamlit
```bash
streamlit run app.py


