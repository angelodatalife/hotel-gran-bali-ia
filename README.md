**Hotel Gran Bali - NeuralClean**

Sistema inteligente de gestión de limpieza que optimiza la asignación de tareas según perfil de cliente.

📊 Descripción

458 habitaciones, 51 plantas, 3 sectores

Predicción de late checkout (ANN)

Estimación de tiempo de limpieza (XGBoost, MAE ~3.2 min)

Segmentación de habitaciones en 6 perfiles (K-Means)

Análisis de sentimiento de opiniones (NLP), mediante Transformer de HF

📁 Modelos

ann.pkl - Red neuronal para late checkout

xgboost.pkl - Regresor de tiempos

kmeans.pkl - Clustering de perfiles

🎯 Resultados

Carga equilibrada entre camareras

Tiempos de limpieza precisos (±3 min)

6 perfiles de cliente identificados

🗒️ Para utilizar la aplicación utiliza los CSV de la carpeta "DATASETS"

<div align="center">
  <a href="https://hotel-gran-bali-ia.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</div>
