# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con pantalla de inicio centralizada y mejoras visuales
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import time
import os
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

st.set_page_config(
    page_title="Hotel Gran Bali - IA Limpieza",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CARGA DE MODELOS
# =============================================================================

@st.cache_resource
def cargar_modelos():
    """Carga todos los modelos necesarios"""
    modelos = {}
    archivos_modelos = {
        'ann': 'ann.pkl',
        'xgboost': 'xgboost.pkl',
        'kmeans': 'kmeans.pkl',
        'nlp': 'nlp.pkl'
    }
    
    for nombre, archivo in archivos_modelos.items():
        if os.path.exists(archivo):
            try:
                with open(archivo, 'rb') as f:
                    modelos[nombre] = pickle.load(f)
            except Exception as e:
                modelos[nombre] = None
        else:
            modelos[nombre] = None
    
    return modelos

modelos = cargar_modelos()

# =============================================================================
# CONSTANTES
# =============================================================================

TOTAL_CAMARERAS = 35
TOTAL_HABITACIONES = 446

# =============================================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# =============================================================================

if 'df_pms' not in st.session_state:
    st.session_state.df_pms = None
if 'incidencias' not in st.session_state:
    st.session_state.incidencias = []  # Falta suministros, Muy sucia, Ocupada, Otro
if 'mantenimiento' not in st.session_state:
    st.session_state.mantenimiento = []  # Averías técnicas
if 'opiniones' not in st.session_state:
    st.session_state.opiniones = []
if 'camarera_actual' not in st.session_state:
    st.session_state.camarera_actual = None
if 'cronometro_activo' not in st.session_state:
    st.session_state.cronometro_activo = False
if 'tiempo_inicio' not in st.session_state:
    st.session_state.tiempo_inicio = None
if 'habitacion_actual' not in st.session_state:
    st.session_state.habitacion_actual = None
if 'asignacion_por_camarera' not in st.session_state:
    st.session_state.asignacion_por_camarera = {}
if 'habitaciones_completadas' not in st.session_state:
    st.session_state.habitaciones_completadas = []
if 'habitaciones_standby' not in st.session_state:
    st.session_state.habitaciones_standby = []  # Lista de IDs en Stand By
if 'reporte_expander_open' not in st.session_state:
    st.session_state.reporte_expander_open = False
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "📊 Gerente"
if 'num_camareras' not in st.session_state:
    st.session_state.num_camareras = TOTAL_CAMARERAS
if 'archivo_cargado' not in st.session_state:
    st.session_state.archivo_cargado = False
if 'cluster_habitaciones' not in st.session_state:
    st.session_state.cluster_habitaciones = {}  # Para almacenar clusters K-Means
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}  # Para codificar variables categóricas

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def limpiar_texto_opinion(texto):
    if not isinstance(texto, str) or texto == "":
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def procesar_opinion(texto):
    """Procesa una opinión y devuelve el sentimiento usando el modelo NLP"""
    if not texto or texto == "":
        return ""
    if modelos.get('nlp') is None:
        return "neutral"
    pipeline = modelos['nlp']
    texto_limpio = limpiar_texto_opinion(texto)
    try:
        return pipeline.predict([texto_limpio])[0]
    except:
        return "neutral"

def formatear_tiempo(segundos):
    minutos = int(segundos // 60)
    segs = int(segundos % 60)
    return f"{minutos}:{segs:02d}"

def aplicar_kmeans(df):
    """Aplica K-Means para segmentar habitaciones por perfil de limpieza"""
    # Verificar si el modelo K-Means existe y es accesible
    if modelos.get('kmeans') is None or df is None or len(df) == 0:
        return {}
    
    try:
        # Extraer el modelo K-Means (puede ser un dict o el modelo directamente)
        kmeans_model = modelos['kmeans']
        
        # Si es un diccionario, intentar extraer el modelo
        if isinstance(kmeans_model, dict):
            if 'modelo' in kmeans_model:
                kmeans_model = kmeans_model['modelo']
            else:
                # Buscar cualquier clave que pueda contener el modelo
                for key in ['kmeans', 'model', 'clusterer']:
                    if key in kmeans_model:
                        kmeans_model = kmeans_model[key]
                        break
                else:
                    # Si no encontramos nada, no podemos usar K-Means
                    return {}
        
        # Verificar que el modelo tiene método predict
        if not hasattr(kmeans_model, 'predict'):
            return {}
        
        df_copy = df.copy()
        
        # Codificar variables categóricas para K-Means
        categorical_cols = ['tipo_habitacion', 'sector', 'segmento']
        for col in categorical_cols:
            if col in df_copy.columns:
                if col not in st.session_state.label_encoders:
                    st.session_state.label_encoders[col] = LabelEncoder()
                    # Fit con valores únicos de esta columna
                    valores_unicos = df_copy[col].astype(str).unique()
                    st.session_state.label_encoders[col].fit(valores_unicos)
                    df_copy[col + '_encoded'] = st.session_state.label_encoders[col].transform(df_copy[col].astype(str))
                else:
                    # Manejar valores nuevos no vistos durante el entrenamiento
                    try:
                        df_copy[col + '_encoded'] = st.session_state.label_encoders[col].transform(df_copy[col].astype(str))
                    except:
                        # Si hay valores nuevos, asignar el valor más frecuente (0)
                        df_copy[col + '_encoded'] = 0
        
        # Seleccionar features para clustering
        feature_cols = []
        if 'planta' in df_copy.columns:
            feature_cols.append('planta')
        if 'noches_estancia' in df_copy.columns:
            feature_cols.append('noches_estancia')
        if 'num_huespedes' in df_copy.columns:
            feature_cols.append('num_huespedes')
        if 'tiene_ninos' in df_copy.columns:
            # Convertir boolean a int
            df_copy['tiene_ninos_int'] = df_copy['tiene_ninos'].astype(int)
            feature_cols.append('tiene_ninos_int')
        if 'tipo_habitacion_encoded' in df_copy.columns:
            feature_cols.append('tipo_habitacion_encoded')
        if 'sector_encoded' in df_copy.columns:
            feature_cols.append('sector_encoded')
        if 'segmento_encoded' in df_copy.columns:
            feature_cols.append('segmento_encoded')
        
        if len(feature_cols) < 2:
            return {}
        
        X = df_copy[feature_cols].values
        
        # Predecir clusters
        clusters = kmeans_model.predict(X)
        
        # Mapear habitación a cluster
        cluster_dict = {}
        for i, hab_id in enumerate(df_copy['habitacion_id'].values):
            cluster_dict[hab_id] = int(clusters[i])
        
        return cluster_dict
    except Exception as e:
        # Silenciosamente fallar sin mostrar warning
        return {}

def predecir_late_checkout_xgboost(df):
    """Usa XGBoost para predecir late checkout"""
    # Verificar si el modelo XGBoost existe
    if modelos.get('xgboost') is None or df is None or len(df) == 0:
        return df
    
    try:
        # Extraer el modelo XGBoost
        xgb_model = modelos['xgboost']
        
        # Si es un diccionario, intentar extraer el modelo
        if isinstance(xgb_model, dict):
            if 'modelo' in xgb_model:
                xgb_model = xgb_model['modelo']
            else:
                # Buscar cualquier clave que pueda contener el modelo
                for key in ['xgb', 'model', 'classifier']:
                    if key in xgb_model:
                        xgb_model = xgb_model[key]
                        break
                else:
                    return df
        
        # Verificar que el modelo tiene método predict_proba
        if not hasattr(xgb_model, 'predict_proba'):
            return df
        
        df_copy = df.copy()
        
        # Preparar features para XGBoost
        feature_cols = []
        
        # Features numéricas
        numeric_features = ['noches_estancia', 'num_huespedes', 'planta']
        for col in numeric_features:
            if col in df_copy.columns:
                feature_cols.append(col)
        
        # Codificar variables categóricas
        categorical_features = ['tipo_habitacion', 'nacionalidad', 'segmento', 'dia_semana']
        for col in categorical_features:
            if col in df_copy.columns:
                if col not in st.session_state.label_encoders:
                    st.session_state.label_encoders[col] = LabelEncoder()
                    valores_unicos = df_copy[col].astype(str).unique()
                    st.session_state.label_encoders[col].fit(valores_unicos)
                    df_copy[col + '_xgb'] = st.session_state.label_encoders[col].transform(df_copy[col].astype(str))
                else:
                    try:
                        df_copy[col + '_xgb'] = st.session_state.label_encoders[col].transform(df_copy[col].astype(str))
                    except:
                        df_copy[col + '_xgb'] = 0
                feature_cols.append(col + '_xgb')
        
        if len(feature_cols) < 2:
            return df
        
        X = df_copy[feature_cols].values
        
        # Predecir probabilidad
        prob_late = xgb_model.predict_proba(X)[:, 1]
        df['prob_late_xgb'] = prob_late
        df['late_checkout_pred_xgb'] = (prob_late > 0.5).astype(int)
        
        # Combinar con ANN si existe para mejor precisión
        if 'prob_late' in df.columns:
            # Promedio ponderado (50% ANN, 50% XGBoost)
            df['prob_late_combinada'] = (df['prob_late'] * 0.5 + prob_late * 0.5)
            df['late_checkout_pred_combinado'] = (df['prob_late_combinada'] > 0.5).astype(int)
        
        return df
    except Exception as e:
        # Silenciosamente fallar sin mostrar warning
        return df

def asignar_por_bloques_adyacentes(df, num_camareras=TOTAL_CAMARERAS):
    """Asigna habitaciones por BLOQUES DE PLANTAS ADYACENTES"""
    if df is None or len(df) == 0:
        return {}
    
    df_asignar = df.copy()
    
    # Aplicar XGBoost para mejorar predicción de late checkout
    df_asignar = predecir_late_checkout_xgboost(df_asignar)
    
    # Aplicar K-Means para obtener clusters
    cluster_dict = aplicar_kmeans(df_asignar)
    st.session_state.cluster_habitaciones = cluster_dict
    if cluster_dict:
        df_asignar['cluster'] = df_asignar['habitacion_id'].map(cluster_dict).fillna(0).astype(int)
    
    plantas_totales = sorted(df_asignar['planta'].unique())
    
    # 1. Calcular carga por planta
    carga_por_planta = {}
    for planta in plantas_totales:
        df_planta = df_asignar[df_asignar['planta'] == planta]
        if 'tiempo_estimado' in df_planta.columns:
            carga_por_planta[planta] = df_planta['tiempo_estimado'].sum()
        else:
            carga_por_planta[planta] = len(df_planta) * 25
    
    # 2. Calcular carga total y carga ideal por camarera
    carga_total = sum(carga_por_planta.values())
    carga_ideal_por_cam = carga_total / num_camareras
    
    # 3. Aplicar ANN para priorizar urgentes (ya existente)
    if modelos.get('ann') is not None:
        try:
            # Extraer el modelo ANN
            ann_data = modelos['ann']
            if isinstance(ann_data, dict):
                ann_model = ann_data.get('modelo')
                scaler_ann = ann_data.get('scaler')
                feature_cols = ann_data.get('feature_cols', [])
            else:
                ann_model = None
                scaler_ann = None
                feature_cols = []
            
            if ann_model is not None and scaler_ann is not None and feature_cols:
                cols_disponibles = [c for c in feature_cols if c in df_asignar.columns]
                if len(cols_disponibles) == len(feature_cols):
                    X_ann = df_asignar[feature_cols].values
                    X_ann_scaled = scaler_ann.transform(X_ann)
                    
                    prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                    df_asignar['prob_late'] = prob_late
                    df_asignar['late_checkout_pred'] = (prob_late > 0.5).astype(int)
                else:
                    df_asignar['late_checkout_pred'] = 0
            else:
                df_asignar['late_checkout_pred'] = 0
        except:
            df_asignar['late_checkout_pred'] = 0
    else:
        df_asignar['late_checkout_pred'] = 0
    
    # Usar la predicción combinada si existe
    if 'late_checkout_pred_combinado' in df_asignar.columns:
        col_prioridad = 'late_checkout_pred_combinado'
    else:
        col_prioridad = 'late_checkout_pred'
    
    # 4. Crear BLOQUES DE PLANTAS ADYACENTES
    bloques = []
    bloque_actual = []
    carga_acumulada = 0
    
    for planta in plantas_totales:
        bloque_actual.append(planta)
        carga_acumulada += carga_por_planta[planta]
        
        if carga_acumulada >= carga_ideal_por_cam * 0.8:
            bloques.append({
                'plantas': bloque_actual.copy(),
                'carga': carga_acumulada
            })
            bloque_actual = []
            carga_acumulada = 0
    
    if bloque_actual:
        bloques.append({
            'plantas': bloque_actual,
            'carga': carga_acumulada
        })
    
    # 5. Asignar camareras a bloques
    asignacion = {}
    cam_idx = 1
    
    for bloque in bloques:
        plantas_bloque = bloque['plantas']
        carga_bloque = bloque['carga']
        
        num_cam_bloque = max(1, round(carga_bloque / carga_ideal_por_cam))
        df_bloque = df_asignar[df_asignar['planta'].isin(plantas_bloque)].copy()
        
        # Ordenar por prioridad (late checkout) y cluster (profundidad de limpieza)
        if 'cluster' in df_bloque.columns:
            df_bloque = df_bloque.sort_values(
                by=[col_prioridad, 'cluster', 'habitacion_id'], 
                ascending=[False, False, True]
            )
        else:
            df_bloque = df_bloque.sort_values(
                by=[col_prioridad, 'habitacion_id'], 
                ascending=[False, True]
            )
        
        if num_cam_bloque > 1:
            habs_por_cam = len(df_bloque) // num_cam_bloque
            resto = len(df_bloque) % num_cam_bloque
            inicio = 0
            for i in range(num_cam_bloque):
                if cam_idx > num_camareras:
                    break
                fin = inicio + habs_por_cam + (1 if i < resto else 0)
                df_cam = df_bloque.iloc[inicio:fin].copy()
                if len(df_cam) > 0:
                    asignacion[f"Camarera {cam_idx:02d}"] = df_cam
                inicio = fin
                cam_idx += 1
        else:
            if cam_idx <= num_camareras:
                asignacion[f"Camarera {cam_idx:02d}"] = df_bloque
                cam_idx += 1
    
    # Asegurar que tenemos exactamente el número de camareras solicitado
    while cam_idx <= num_camareras and len(asignacion) < num_camareras:
        if asignacion:
            # Tomar la camarera con más habitaciones y dividir
            cam_max = max(asignacion.items(), key=lambda x: len(x[1]))
            df_max = cam_max[1]
            if len(df_max) > 1:
                mitad = len(df_max) // 2
                df_cam1 = df_max.iloc[:mitad].copy()
                df_cam2 = df_max.iloc[mitad:].copy()
                
                asignacion[cam_max[0]] = df_cam1
                if len(df_cam2) > 0:
                    asignacion[f"Camarera {cam_idx:02d}"] = df_cam2
                    cam_idx += 1
            else:
                break
        else:
            break
    
    return asignacion

def actualizar_dataset(hab_id, campo, valor):
    """Actualiza una columna específica en el dataset para una habitación"""
    if st.session_state.df_pms is not None:
        df = st.session_state.df_pms
        if hab_id in df['habitacion_id'].values:
            df.loc[df['habitacion_id'] == hab_id, campo] = valor
            st.session_state.df_pms = df

def procesar_archivo(archivo):
    """Procesa el archivo cargado y actualiza el estado de la sesión"""
    with st.spinner("Procesando archivo..."):
        df = pd.read_csv(archivo)
        
        # Asegurar que existen las columnas necesarias
        columnas_necesarias = ['tiempo_real', 'incidencia_camarera', 'opinion_cliente', 'sentimiento_nlp']
        for col in columnas_necesarias:
            if col not in df.columns:
                df[col] = None
        
        # Aplicar ANN
        if modelos.get('ann') is not None:
            try:
                ann_data = modelos['ann']
                if isinstance(ann_data, dict):
                    ann_model = ann_data.get('modelo')
                    scaler_ann = ann_data.get('scaler')
                    feature_cols = ann_data.get('feature_cols', [])
                else:
                    ann_model = None
                    scaler_ann = None
                    feature_cols = []
                
                if ann_model is not None and scaler_ann is not None and feature_cols:
                    cols_disponibles = [c for c in feature_cols if c in df.columns]
                    if len(cols_disponibles) == len(feature_cols):
                        X_ann = df[feature_cols].values
                        X_ann_scaled = scaler_ann.transform(X_ann)
                        
                        prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                        df['prob_late'] = prob_late
                        df['late_checkout_pred'] = (prob_late > 0.5).astype(int)
            except Exception as e:
                pass  # Silenciosamente ignorar errores de ANN
        
        st.session_state.df_pms = df
        
        with st.spinner("Calculando asignación por bloques adyacentes..."):
            st.session_state.asignacion_por_camarera = asignar_por_bloques_adyacentes(df, st.session_state.num_camareras)
        
        st.session_state.archivo_cargado = True
        st.session_state.selected_page = "📊 Gerente"
        st.success(f"✅ PMS cargado: {len(df)} habitaciones")
        time.sleep(1)
        st.rerun()

# =============================================================================
# PANTALLA DE INICIO (antes de cargar archivo)
# =============================================================================

def mostrar_pantalla_inicio():
    """Muestra la pantalla de inicio centralizada"""
    
    # Título principal (más pequeño y combinado)
    st.markdown(
        """
        <h2 style='text-align: center; color: #1E88E5; margin-top: 50px; margin-bottom: 30px;'>
            🏨 Hotel Gran Bali - Sistema de Gestión IA de Limpieza
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    # Contenedor central para la carga de archivos
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Título "Cargar PMS" sin recuadro
        st.markdown(
            """
            <h3 style='text-align: center; color: #333; margin-bottom: 20px;'>
                Cargar PMS aquí ⬇️
            </h3>
            """,
            unsafe_allow_html=True
        )
        
        # File uploader de Streamlit (estilo drag and drop)
        archivo = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type=['csv'],
            key="file_uploader_inicio",
            label_visibility="collapsed"
        )
        
        if archivo is not None:
            procesar_archivo(archivo)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Sección de modelos cargados (centrada)
        st.markdown(
            """
            <h4 style='text-align: center; color: #666; margin-bottom: 20px;'>
                🤖 Modelos cargados
            </h4>
            """,
            unsafe_allow_html=True
        )
        
        # Mostrar modelos en fila (centrados)
        col_mod1, col_mod2, col_mod3, col_mod4 = st.columns(4)
        
        with col_mod1:
            if modelos.get('ann') is not None:
                st.markdown(
                    """
                    <div style="
                        background-color: #e8f5e8;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>✅ ANN</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color: #ffebee;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>❌ ANN</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col_mod2:
            if modelos.get('xgboost') is not None:
                st.markdown(
                    """
                    <div style="
                        background-color: #e8f5e8;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>✅ XGBoost</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color: #ffebee;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>❌ XGBoost</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col_mod3:
            if modelos.get('kmeans') is not None:
                st.markdown(
                    """
                    <div style="
                        background-color: #e8f5e8;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>✅ K-Means</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color: #ffebee;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>❌ K-Means</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col_mod4:
            if modelos.get('nlp') is not None:
                st.markdown(
                    """
                    <div style="
                        background-color: #e8f5e8;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>✅ NLP</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color: #ffebee;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <strong>❌ NLP</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =============================================================================
# SIDEBAR - NAVEGACIÓN (solo visible después de cargar archivo)
# =============================================================================

def mostrar_sidebar():
    """Muestra el sidebar con la navegación (solo después de cargar archivo)"""
    with st.sidebar:
        st.title("🏨 Hotel Gran Bali")
        st.markdown("---")
        
        # Menú principal con contadores alineados
        st.markdown("**Menú Principal**")
        
        # Gerente
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            gerente_selected = st.button("📊 Gerente", key="btn_gerente", use_container_width=True)
        with col2:
            st.markdown("")
        with col3:
            st.markdown("")
        
        # Camarera
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            camarera_selected = st.button("🧹 Camarera", key="btn_camarera", use_container_width=True)
        with col2:
            st.markdown("")
        with col3:
            st.markdown("")
        
        # Incidencias con contador
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            incidencias_selected = st.button("⚠️ Incidencias", key="btn_incidencias", use_container_width=True)
        with col2:
            if st.session_state.incidencias:
                st.markdown(f"**({len(st.session_state.incidencias)})**")
        with col3:
            st.markdown("")
        
        # Mantenimiento con contador
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            mantenimiento_selected = st.button("🔧 Mantenimiento", key="btn_mantenimiento", use_container_width=True)
        with col2:
            if st.session_state.mantenimiento:
                st.markdown(f"**({len(st.session_state.mantenimiento)})**")
        with col3:
            st.markdown("")
        
        # Cliente
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            cliente_selected = st.button("👤 Cliente", key="btn_cliente", use_container_width=True)
        with col2:
            st.markdown("")
        with col3:
            st.markdown("")
        
        # Dataset
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            dataset_selected = st.button("📋 Dataset", key="btn_dataset", use_container_width=True)
        with col2:
            st.markdown("")
        with col3:
            st.markdown("")
        
        # Determinar la página seleccionada
        if gerente_selected:
            st.session_state.selected_page = "📊 Gerente"
        elif camarera_selected:
            st.session_state.selected_page = "🧹 Camarera"
        elif incidencias_selected:
            st.session_state.selected_page = "⚠️ Incidencias"
        elif mantenimiento_selected:
            st.session_state.selected_page = "🔧 Mantenimiento"
        elif cliente_selected:
            st.session_state.selected_page = "👤 Cliente"
        elif dataset_selected:
            st.session_state.selected_page = "📋 Dataset"
        
        st.markdown("---")
        
        # Modelos cargados
        st.subheader("🤖 Modelos cargados")
        modelos_lista = {
            'ANN': modelos.get('ann') is not None,
            'XGBoost': modelos.get('xgboost') is not None,
            'K-Means': modelos.get('kmeans') is not None,
            'NLP': modelos.get('nlp') is not None
        }
        for nombre, cargado in modelos_lista.items():
            if cargado:
                st.markdown(f"✅ {nombre}")
            else:
                st.markdown(f"❌ {nombre}")
        
        st.markdown("---")
        
        if st.button("🔄 Reiniciar Simulación", use_container_width=True):
            for key in ['df_pms', 'incidencias', 'mantenimiento', 'opiniones', 'camarera_actual', 
                        'cronometro_activo', 'tiempo_inicio', 'habitacion_actual',
                        'asignacion_por_camarera', 'habitaciones_completadas', 'habitaciones_standby', 
                        'archivo_cargado', 'cluster_habitaciones', 'label_encoders']:
                if key in st.session_state:
                    if key in ['incidencias', 'mantenimiento', 'opiniones', 'habitaciones_completadas', 
                               'habitaciones_standby']:
                        st.session_state[key] = []
                    elif key in ['cluster_habitaciones', 'label_encoders']:
                        st.session_state[key] = {}
                    else:
                        st.session_state[key] = None
            st.session_state.archivo_cargado = False
            st.rerun()

# =============================================================================
# LÓGICA PRINCIPAL DE NAVEGACIÓN
# =============================================================================

# Si no hay archivo cargado, mostrar pantalla de inicio
if not st.session_state.archivo_cargado or st.session_state.df_pms is None:
    mostrar_pantalla_inicio()
else:
    # Mostrar sidebar y luego la vista correspondiente
    mostrar_sidebar()
    selected = st.session_state.selected_page

# =============================================================================
# VISTA GERENTE - CON 3 PESTAÑAS INTERNAS
# =============================================================================

if st.session_state.archivo_cargado and selected == "📊 Gerente":
    
    df = st.session_state.df_pms
    
    # Crear las 3 pestañas internas
    tab_dashboard, tab_estado, tab_carga = st.tabs(["📊 Dashboard Gerente", "🗺️ Estado de Habitaciones", "📊 Carga de trabajo por camarera"])
    
    # ===== PESTAÑA 1: DASHBOARD GERENTE (CÍRCULOS Y CONTROLES) =====
    with tab_dashboard:
        st.title("📊 Dashboard Gerente - Hotel Gran Bali")
        
        # Métricas principales en círculos
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            ocupacion = len(df) / TOTAL_HABITACIONES * 100
            st.markdown(
                f"""
                <div style="
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    border: 4px solid #1E88E5;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    background: transparent;
                ">
                    <div style="font-size: 32px; font-weight: bold;">{ocupacion:.1f}%</div>
                    <div style="font-size: 16px;">Ocupación</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_metric2:
            habitaciones_hechas = len(st.session_state.habitaciones_completadas)
            st.markdown(
                f"""
                <div style="
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    border: 4px solid #4CAF50;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    background: transparent;
                ">
                    <div style="font-size: 32px; font-weight: bold;">{habitaciones_hechas}/{len(df)}</div>
                    <div style="font-size: 16px;">Habitaciones</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_metric3:
            st.markdown(
                f"""
                <div style="
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    border: 4px solid #FFA500;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    background: transparent;
                ">
                    <div style="font-size: 32px; font-weight: bold;">{st.session_state.num_camareras}</div>
                    <div style="font-size: 16px;">Camareras</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Selector de número de camareras
        col_control1, col_control2, col_control3 = st.columns([1, 2, 1])
        with col_control2:
            st.markdown("### Ajustar personal")
            col_plus, col_num, col_minus = st.columns([1, 2, 1])
            with col_plus:
                if st.button("➕", key="btn_plus"):
                    st.session_state.num_camareras = min(50, st.session_state.num_camareras + 1)
                    st.session_state.asignacion_por_camarera = asignar_por_bloques_adyacentes(
                        st.session_state.df_pms, 
                        st.session_state.num_camareras
                    )
                    st.rerun()
            with col_num:
                st.markdown(f"<h2 style='text-align: center;'>{st.session_state.num_camareras}</h2>", unsafe_allow_html=True)
            with col_minus:
                if st.button("➖", key="btn_minus"):
                    st.session_state.num_camareras = max(1, st.session_state.num_camareras - 1)
                    st.session_state.asignacion_por_camarera = asignar_por_bloques_adyacentes(
                        st.session_state.df_pms, 
                        st.session_state.num_camareras
                    )
                    st.rerun()
        
        st.markdown("---")
        
        # Información adicional
        st.subheader("📈 Resumen rápido")
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            # Usar la mejor predicción disponible
            if 'late_checkout_pred_combinado' in df.columns:
                checkouts = int(df['late_checkout_pred_combinado'].sum())
                st.metric("Checkouts estimados (combinado)", checkouts)
            elif 'late_checkout_pred' in df.columns:
                st.metric("Checkouts estimados (ANN)", int(df['late_checkout_pred'].sum()))
            elif 'late_checkout_pred_xgb' in df.columns:
                st.metric("Checkouts estimados (XGBoost)", int(df['late_checkout_pred_xgb'].sum()))
            else:
                st.metric("Checkouts estimados", 0)
        with col_res2:
            st.metric("Repasos", len(df) - (int(df['late_checkout_pred'].sum()) if 'late_checkout_pred' in df.columns else 0))
        with col_res3:
            st.metric("Incidencias", len(st.session_state.incidencias))
        with col_res4:
            st.metric("Mantenimiento", len(st.session_state.mantenimiento))
        
        # Información de modelos utilizados
        st.subheader("🤖 Modelos activos en esta sesión")
        col_mod1, col_mod2, col_mod3, col_mod4 = st.columns(4)
        with col_mod1:
            if modelos.get('ann') is not None:
                st.success("✅ ANN: Prediciendo late checkout")
            else:
                st.warning("❌ ANN no disponible")
        with col_mod2:
            if modelos.get('xgboost') is not None:
                st.success("✅ XGBoost: Mejorando predicciones")
            else:
                st.warning("❌ XGBoost no disponible")
        with col_mod3:
            if modelos.get('kmeans') is not None and st.session_state.cluster_habitaciones:
                num_clusters = len(set(st.session_state.cluster_habitaciones.values()))
                st.success(f"✅ K-Means: {num_clusters} perfiles de limpieza")
            else:
                st.warning("❌ K-Means no disponible")
        with col_mod4:
            if modelos.get('nlp') is not None:
                st.success("✅ NLP: Análisis de opiniones")
            else:
                st.warning("❌ NLP no disponible")
    
    # ===== PESTAÑA 2: ESTADO DE HABITACIONES (MAPA DE COLORES) =====
    with tab_estado:
        st.title("🗺️ Estado de Habitaciones")
        
        # Definir sectores
        sectores = {
            'Bajo (Pl 2-15)': list(range(2, 16)),
            'Medio (Pl 16-30)': list(range(16, 31)),
            'Alto (Pl 31-52)': list(range(31, 53))
        }
        
        # Función para determinar el color de cada habitación
        def get_room_color(hab_id):
            if hab_id in st.session_state.habitaciones_completadas:
                # Buscar si tuvo incidencia
                for inc in st.session_state.incidencias:
                    if inc['habitacion'] == hab_id:
                        if inc['tipo'] == "Falta suministros":
                            return "#FFA500"  # Naranja
                        elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                            return "#FF4444"  # Rojo
                for mant in st.session_state.mantenimiento:
                    if mant['habitacion'] == hab_id:
                        return "#808080"  # Gris
                return "#4CAF50"  # Verde (completada sin problemas)
            elif hab_id in st.session_state.habitaciones_standby:
                # Buscar el tipo de problema para el color correcto
                for inc in st.session_state.incidencias:
                    if inc['habitacion'] == hab_id:
                        if inc['tipo'] == "Falta suministros":
                            return "#FFA500"  # Naranja
                        elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                            return "#FF4444"  # Rojo
                return "#FFA500"  # Naranja por defecto para standby
            return "#FFFFFF"  # Blanco (pendiente)
        
        # Crear pestañas por sector
        subtab1, subtab2, subtab3 = st.tabs(["🔵 Sector Bajo", "🟡 Sector Medio", "🔴 Sector Alto"])
        
        for subtab, (sector_nombre, plantas) in zip([subtab1, subtab2, subtab3], sectores.items()):
            with subtab:
                # Filtrar habitaciones del sector
                df_sector = df[df['planta'].isin(plantas)]
                
                if len(df_sector) == 0:
                    st.info(f"No hay habitaciones en {sector_nombre}")
                    continue
                
                # Ordenar por número de habitación
                df_sector = df_sector.sort_values('habitacion_id')
                
                # Crear grid de habitaciones
                cols_por_fila = 6
                habitaciones = df_sector['habitacion_id'].tolist()
                
                for i in range(0, len(habitaciones), cols_por_fila):
                    cols = st.columns(cols_por_fila)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(habitaciones):
                            hab_id = habitaciones[idx]
                            color = get_room_color(hab_id)
                            
                            # Buscar información adicional
                            tooltip = f"Hab {hab_id}"
                            row = df_sector[df_sector['habitacion_id'] == hab_id].iloc[0]
                            if 'tiempo_estimado' in row:
                                tooltip += f"\nTiempo: {row['tiempo_estimado']} min"
                            if st.session_state.cluster_habitaciones and hab_id in st.session_state.cluster_habitaciones:
                                cluster = st.session_state.cluster_habitaciones[hab_id]
                                tooltip += f"\nPerfil: {cluster}"
                            
                            # Cuadrado con número de habitación
                            col.markdown(
                                f"""
                                <div style="
                                    background-color: {color};
                                    border: 2px solid #ddd;
                                    border-radius: 8px;
                                    padding: 10px;
                                    margin: 5px;
                                    text-align: center;
                                    font-weight: bold;
                                    cursor: help;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                " title="{tooltip}">
                                    {int(hab_id)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            col.markdown("")  # Celda vacía
        
        st.markdown("---")
        
        # Leyenda de colores
        st.subheader("📋 Leyenda")
        col_leg1, col_leg2, col_leg3, col_leg4, col_leg5 = st.columns(5)
        
        with col_leg1:
            st.markdown(
                """
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #FFFFFF; border: 2px solid #ddd; border-radius: 4px; margin-right: 8px;"></div>
                    <span>Pendiente</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_leg2:
            st.markdown(
                """
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #4CAF50; border-radius: 4px; margin-right: 8px;"></div>
                    <span>Completada</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_leg3:
            st.markdown(
                """
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #808080; border-radius: 4px; margin-right: 8px;"></div>
                    <span>Mantenimiento</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_leg4:
            st.markdown(
                """
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #FFA500; border-radius: 4px; margin-right: 8px;"></div>
                    <span>Falta suministros</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_leg5:
            st.markdown(
                """
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #FF4444; border-radius: 4px; margin-right: 8px;"></div>
                    <span>Muy sucia / Ocupada</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # ===== PESTAÑA 3: CARGA DE TRABAJO POR CAMARERA =====
    with tab_carga:
        st.title("📊 Carga de trabajo por camarera")
        
        if st.session_state.asignacion_por_camarera:
            # Definir sectores
            sectores_carga = {
                'Sector Bajo': list(range(2, 16)),
                'Sector Medio': list(range(16, 31)),
                'Sector Alto': list(range(31, 53))
            }
            
            # Crear gráficos por sector
            for sector_nombre, plantas in sectores_carga.items():
                st.markdown(f"### {sector_nombre}")
                
                # Filtrar camareras que trabajan en este sector
                datos_sector = []
                for i in range(1, st.session_state.num_camareras + 1):
                    cam = f"Camarera {i:02d}"
                    if cam in st.session_state.asignacion_por_camarera:
                        df_cam = st.session_state.asignacion_por_camarera[cam]
                        # Verificar si la camarera trabaja en este sector
                        if any(p in plantas for p in df_cam['planta'].unique()):
                            num_hab = len(df_cam)
                            datos_sector.append({
                                'Camarera': cam,
                                'Habitaciones': num_hab
                            })
                
                if datos_sector:
                    df_carga_sector = pd.DataFrame(datos_sector)
                    media_hab = df_carga_sector['Habitaciones'].mean()
                    umbral_alto = media_hab * 1.2
                    
                    # Crear gráfico de barras para el sector
                    fig = go.Figure()
                    
                    for _, row in df_carga_sector.iterrows():
                        color = '#FF4444' if row['Habitaciones'] > umbral_alto else '#1E88E5'
                        fig.add_trace(go.Bar(
                            x=[row['Camarera']],
                            y=[row['Habitaciones']],
                            name=row['Camarera'],
                            marker_color=color,
                            showlegend=False
                        ))
                    
                    fig.add_hline(
                        y=media_hab,
                        line_dash="dash",
                        line_color="blue",
                        annotation_text=f"Media: {media_hab:.1f}",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        title=f'Carga de trabajo - {sector_nombre}',
                        xaxis_tickangle=-45,
                        yaxis_title="Número de habitaciones",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar camareras con sobrecarga en este sector
                    sobrecargadas = df_carga_sector[df_carga_sector['Habitaciones'] > umbral_alto]
                    if len(sobrecargadas) > 0:
                        st.warning(f"⚠️ Camareras con sobrecarga en {sector_nombre}:")
                        for _, row in sobrecargadas.iterrows():
                            st.markdown(f"- {row['Camarera']}: {row['Habitaciones']} habitaciones")
                else:
                    st.info(f"No hay camareras asignadas al sector {sector_nombre}")
                
                st.markdown("---")
            
            # Añadir información de clusters si K-Means está disponible
            if modelos.get('kmeans') is not None and st.session_state.cluster_habitaciones:
                st.subheader("📊 Perfiles de limpieza por sector (K-Means)")
                
                # Crear DataFrame con clusters
                df_clusters = df.copy()
                df_clusters['cluster'] = df_clusters['habitacion_id'].map(st.session_state.cluster_habitaciones).fillna(0)
                
                # Distribución de clusters por sector
                for sector_nombre, plantas in sectores_carga.items():
                    df_sector = df_clusters[df_clusters['planta'].isin(plantas)]
                    if len(df_sector) > 0:
                        cluster_counts = df_sector['cluster'].value_counts().sort_index()
                        
                        # Crear gráfico de distribución
                        fig = px.pie(
                            values=cluster_counts.values,
                            names=[f"Perfil {i}" for i in cluster_counts.index],
                            title=f"Distribución de perfiles - {sector_nombre}",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explicación de perfiles
                        st.caption("""
                        **Perfiles de limpieza:**
                        - **Perfil 0**: Limpieza rápida (habitaciones estándar, pocas noches)
                        - **Perfil 1**: Limpieza media (familias, estancias medias)
                        - **Perfil 2**: Limpieza profunda (suites, VIP, muchas noches)
                        """)
        else:
            st.info("No hay datos de asignación disponibles")

# =============================================================================
# VISTA CAMARERA (solo si hay archivo cargado)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "🧹 Camarera":
    st.title("🧹 App Camarera - Hotel Gran Bali")
    
    if not st.session_state.asignacion_por_camarera:
        st.warning("⚠️ No hay asignación disponible")
    else:
        if st.session_state.camarera_actual is None:
            st.subheader("👤 Selecciona tu perfil")
            
            # Selector con dirección hacia abajo
            camareras = [f"Camarera {i:02d}" for i in range(1, st.session_state.num_camareras + 1)]
            st.session_state.camarera_actual = st.selectbox(
                "Nombre:",
                camareras,
                index=None,
                placeholder="Elige tu nombre...",
                key="select_camarera_principal",
                label_visibility="collapsed"
            )
            
            if st.session_state.camarera_actual:
                st.session_state.habitaciones_completadas = []
                st.session_state.habitaciones_standby = []
                st.rerun()
        else:
            # Verificar si hay una limpieza en curso
            limpieza_en_curso = st.session_state.cronometro_activo and st.session_state.habitacion_actual is not None
            
            # Obtener asignación de esta camarera
            df_asignadas = st.session_state.asignacion_por_camarera.get(
                st.session_state.camarera_actual, 
                pd.DataFrame()
            )
            
            total_asignadas = len(df_asignadas)
            completadas = len(st.session_state.habitaciones_completadas)
            pendientes = total_asignadas - completadas - len(st.session_state.habitaciones_standby)
            
            # Mostrar información de la camarera
            col_info1, col_info2, col_info3, col_info4 = st.columns([2, 2, 2, 1])
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                if len(df_asignadas) > 0:
                    plantas_unicas = sorted(df_asignadas['planta'].unique())
                    st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)}")
            with col_info3:
                # Botón sin parámetro disabled para evitar errores
                if st.button("🔄 Cambiar usuario", key="btn_cambiar_usuario_principal"):
                    st.session_state.camarera_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.session_state.habitaciones_standby = []
                    st.rerun()
                if limpieza_en_curso:
                    st.caption("⚠️ Finaliza la limpieza actual para cambiar")
            with col_info4:
                if st.session_state.habitaciones_standby:
                    st.markdown(f"**⏸️ Stand By:** {len(st.session_state.habitaciones_standby)}")
            
            st.markdown("---")
            
            # Barra de progreso
            if total_asignadas > 0:
                progreso_total = completadas / total_asignadas
                st.progress(progreso_total, text=f"**{completadas}/{total_asignadas}** habitaciones completadas")
            
            # ===== SECCIÓN 1: LIMPIEZA EN CURSO =====
            if limpieza_en_curso:
                with st.container():
                    st.subheader("⏱️ Limpieza en curso")
                    
                    hab = st.session_state.habitacion_actual
                    
                    col_crono1, col_crono2, col_crono3 = st.columns(3)
                    
                    with col_crono1:
                        st.markdown(f"**Habitación:** {int(hab['habitacion_id'])}")
                        st.markdown(f"**Planta:** {int(hab['planta'])}")
                        # Mostrar perfil si está disponible
                        if st.session_state.cluster_habitaciones and hab['habitacion_id'] in st.session_state.cluster_habitaciones:
                            cluster = st.session_state.cluster_habitaciones[hab['habitacion_id']]
                            if cluster == 0:
                                st.markdown("**Perfil:** ⚡ Rápido")
                            elif cluster == 1:
                                st.markdown("**Perfil:** 📊 Estándar")
                            elif cluster == 2:
                                st.markdown("**Perfil:** 🔬 Profundo")
                    
                    with col_crono2:
                        tiempo_transcurrido = (datetime.now() - st.session_state.tiempo_inicio).seconds
                        minutos = tiempo_transcurrido // 60
                        segundos = tiempo_transcurrido % 60
                        st.markdown(f"**Tiempo:** {minutos}:{segundos:02d}")
                        if 'tiempo_estimado' in hab:
                            progreso = min(tiempo_transcurrido / (hab['tiempo_estimado'] * 60), 1.0)
                            st.progress(progreso)
                    
                    with col_crono3:
                        if st.button("✅ Finalizar limpieza", type="primary", use_container_width=True, key="btn_finalizar_principal"):
                            tiempo_real = (datetime.now() - st.session_state.tiempo_inicio).seconds / 60
                            
                            # Actualizar dataset con tiempo_real
                            hab_id = hab['habitacion_id']
                            actualizar_dataset(hab_id, 'tiempo_real', round(tiempo_real, 1))
                            
                            st.session_state.habitaciones_completadas.append(hab_id)
                            
                            st.success(f"✅ Habitación {int(hab_id)} completada en {tiempo_real:.1f} minutos")
                            
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            st.session_state.reporte_expander_open = False
                            time.sleep(1)
                            st.rerun()
                    
                    # Reportar problema
                    with st.expander("⚠️ Reportar problema", expanded=st.session_state.get('reporte_expander_open', False)):
                        tipo_reporte = st.selectbox(
                            "Tipo de problema",
                            ["Mantenimiento (avería)", "Falta suministros", "Muy sucia", "Ocupada", "Otro"],
                            key="tipo_reporte_cron"
                        )
                        desc_reporte = st.text_area("Descripción", key="desc_reporte_cron")
                        
                        col_rep1, col_rep2 = st.columns(2)
                        with col_rep1:
                            if st.button("Enviar y continuar", key="btn_reporte_continuar"):
                                hab_id = hab['habitacion_id']
                                
                                # Guardar en el dataset
                                actualizar_dataset(hab_id, 'incidencia_camarera', f"{tipo_reporte}: {desc_reporte}")
                                
                                # Mantenimiento: solo notifica, no afecta al cronómetro
                                if tipo_reporte == "Mantenimiento (avería)":
                                    st.session_state.mantenimiento.append({
                                        'habitacion': int(hab_id),
                                        'planta': int(hab['planta']),
                                        'tipo': "Avería",
                                        'descripcion': desc_reporte,
                                        'timestamp': datetime.now().strftime("%H:%M"),
                                        'fecha': datetime.now().strftime("%d/%m/%Y"),
                                        'reportado_por': st.session_state.camarera_actual
                                    })
                                    st.success("🔧 Reporte enviado a Mantenimiento")
                                    st.session_state.reporte_expander_open = False
                                    st.rerun()
                                
                                # Incidencias: van a Stand By
                                else:
                                    # Mover a Stand By
                                    st.session_state.habitaciones_standby.append(hab_id)
                                    
                                    # Guardar en incidencias
                                    st.session_state.incidencias.append({
                                        'habitacion': int(hab_id),
                                        'planta': int(hab['planta']),
                                        'tipo': tipo_reporte,
                                        'descripcion': desc_reporte,
                                        'timestamp': datetime.now().strftime("%H:%M"),
                                        'fecha': datetime.now().strftime("%d/%m/%Y"),
                                        'reportado_por': st.session_state.camarera_actual
                                    })
                                    
                                    st.warning(f"⏸️ Habitación {int(hab_id)} movida a Stand By")
                                    
                                    # Reiniciar cronómetro (para nueva habitación)
                                    st.session_state.cronometro_activo = False
                                    st.session_state.habitacion_actual = None
                                    st.session_state.reporte_expander_open = False
                                    time.sleep(1)
                                    st.rerun()
                        
                        with col_rep2:
                            if st.button("Cancelar", key="btn_reporte_cancelar"):
                                st.session_state.reporte_expander_open = False
                                st.rerun()
                    
                    st.markdown("---")
            
            # ===== SECCIÓN 2: HABITACIONES EN STAND BY =====
            if st.session_state.habitaciones_standby:
                with st.container():
                    st.subheader("⏸️ Stand By")
                    st.caption("Habitaciones pendientes por problemas")
                    
                    # Crear una copia de la lista para iterar
                    standby_list = st.session_state.habitaciones_standby.copy()
                    for hab_id in standby_list:
                        row = df_asignadas[df_asignadas['habitacion_id'] == hab_id]
                        if len(row) > 0:
                            row = row.iloc[0]
                            with st.container():
                                cols = st.columns([3, 2, 2, 2])
                                with cols[0]:
                                    st.markdown(f"⏸️ **Hab {int(hab_id)}**")
                                    st.caption(f"Planta {int(row['planta'])}")
                                with cols[1]:
                                    # Buscar la incidencia asociada
                                    inc = next((i for i in st.session_state.incidencias if i['habitacion'] == hab_id), None)
                                    if inc:
                                        st.markdown(f"**{inc['tipo']}**")
                                with cols[2]:
                                    if 'tiempo_estimado' in row:
                                        st.markdown(f"⏱️ {row['tiempo_estimado']} min")
                                with cols[3]:
                                    if st.button("✅ Resuelto", key=f"btn_standby_{hab_id}"):
                                        # Mover a completadas
                                        st.session_state.habitaciones_completadas.append(hab_id)
                                        st.session_state.habitaciones_standby.remove(hab_id)
                                        st.rerun()
                                st.divider()
            
            # ===== SECCIÓN 3: HABITACIONES PENDIENTES =====
            if total_asignadas > 0:
                df_pendientes = df_asignadas[
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_completadas) &
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_standby)
                ].copy()
                
                # Añadir información de cluster para ordenación
                if st.session_state.cluster_habitaciones:
                    df_pendientes['cluster'] = df_pendientes['habitacion_id'].map(st.session_state.cluster_habitaciones).fillna(0)
                
                # Ordenar por prioridad (cluster más alto primero = limpieza profunda)
                if 'cluster' in df_pendientes.columns and 'late_checkout_pred' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['late_checkout_pred', 'cluster', 'habitacion_id'], 
                        ascending=[False, False, False]
                    )
                elif 'cluster' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['cluster', 'habitacion_id'], 
                        ascending=[False, False]
                    )
                elif 'late_checkout_pred' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['late_checkout_pred', 'habitacion_id'], 
                        ascending=[False, False]
                    )
                
                st.markdown(f"### Pendientes ({pendientes} restantes)")
                
                if pendientes == 0 and len(st.session_state.habitaciones_standby) == 0:
                    st.success("🎉 ¡Has completado todas tus habitaciones!")
                    st.balloons()
                else:
                    for i, (_, row) in enumerate(df_pendientes.iterrows()):
                        with st.container():
                            cols = st.columns([3, 2, 2, 3])
                            
                            with cols[0]:
                                if 'late_checkout_pred' in row and row['late_checkout_pred'] == 1:
                                    tipo_emoji = "🔴"
                                else:
                                    tipo_emoji = "🟡"
                                st.markdown(f"{tipo_emoji} **Hab {int(row['habitacion_id'])}**")
                                st.caption(f"Planta {int(row['planta'])}")
                                
                                # Mostrar perfil si está disponible
                                if st.session_state.cluster_habitaciones and row['habitacion_id'] in st.session_state.cluster_habitaciones:
                                    cluster = st.session_state.cluster_habitaciones[row['habitacion_id']]
                                    if cluster == 2:
                                        st.caption("🔬 Limpieza profunda")
                                    elif cluster == 1:
                                        st.caption("📊 Limpieza media")
                                    else:
                                        st.caption("⚡ Limpieza rápida")
                            
                            with cols[1]:
                                if 'late_checkout_pred' in row and row['late_checkout_pred'] == 1:
                                    st.markdown("🏃 **Late**")
                                else:
                                    st.markdown("🛏️ **Normal**")
                            
                            with cols[2]:
                                if 'tiempo_estimado' in row:
                                    st.markdown(f"⏱️ **{row['tiempo_estimado']} min**")
                            
                            with cols[3]:
                                # Botón simple sin disabled
                                if not limpieza_en_curso:
                                    if st.button(
                                        f"▶️ Iniciar", 
                                        key=f"btn_iniciar_{row['habitacion_id']}_{i}",
                                        use_container_width=True
                                    ):
                                        st.session_state.habitacion_actual = row
                                        st.session_state.cronometro_activo = True
                                        st.session_state.tiempo_inicio = datetime.now()
                                        st.session_state.reporte_expander_open = False
                                        st.rerun()
                                else:
                                    st.button(
                                        f"⏸️ En curso", 
                                        key=f"btn_disabled_{row['habitacion_id']}_{i}",
                                        disabled=True,
                                        use_container_width=True
                                    )
                            
                            st.divider()
            
            # ===== SECCIÓN 4: HABITACIONES COMPLETADAS =====
            if completadas > 0:
                st.markdown("---")
                st.subheader(f"✅ Completadas ({completadas}/{total_asignadas})")
                
                df_completadas = st.session_state.df_pms[
                    st.session_state.df_pms['habitacion_id'].isin(
                        st.session_state.habitaciones_completadas
                    )
                ]
                
                for i, (_, row) in enumerate(df_completadas.iterrows()):
                    with st.container():
                        cols = st.columns([3, 2, 2, 3])
                        
                        with cols[0]:
                            st.markdown(f"✅ ~~Hab {int(row['habitacion_id'])}~~")
                            st.caption(f"Planta {int(row['planta'])}")
                        
                        with cols[1]:
                            if 'late_checkout_pred' in row and row['late_checkout_pred'] == 1:
                                st.markdown("~~Late~~")
                            else:
                                st.markdown("~~Normal~~")
                        
                        with cols[2]:
                            if 'tiempo_estimado' in row:
                                st.markdown(f"~~{row['tiempo_estimado']} min~~")
                        
                        with cols[3]:
                            if 'tiempo_real' in row and pd.notna(row['tiempo_real']):
                                st.markdown(f"✅ Real: {row['tiempo_real']} min")
                            else:
                                st.markdown("✅ Listo")
                        
                        st.divider()

# =============================================================================
# VISTA CLIENTE (solo si hay archivo cargado)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "👤 Cliente":
    st.title("👤 Opinión de Clientes")
    st.caption("Comparte tu experiencia para ayudarnos a mejorar")
    
    df = st.session_state.df_pms
    
    with st.form("formulario_opinion_cliente"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Selector de habitación
            habitaciones_disponibles = sorted(df['habitacion_id'].tolist())
            habitacion = st.selectbox(
                "Número de habitación:",
                habitaciones_disponibles,
                index=None,
                placeholder="Selecciona tu habitación"
            )
        
        with col2:
            st.markdown("### 📝 Tu opinión")
            st.markdown("Con respecto a la habitación:")
        
        # Pregunta principal
        st.markdown("#### ¿En qué podríamos mejorar?")
        opinion_texto = st.text_area(
            "Escribe tu opinión aquí:",
            placeholder="Ej: La habitación estaba muy limpia, pero el aire acondicionado hacía ruido...",
            height=150
        )
        
        # Botón de envío
        submitted = st.form_submit_button("📤 Enviar opinión", use_container_width=True, type="primary")
        
        if submitted:
            if not habitacion:
                st.error("❌ Por favor, selecciona tu número de habitación")
            elif not opinion_texto:
                st.error("❌ Por favor, escribe tu opinión")
            else:
                # Procesar la opinión con NLP
                sentimiento = procesar_opinion(opinion_texto)
                
                # Guardar en el dataset
                actualizar_dataset(habitacion, 'opinion_cliente', opinion_texto)
                actualizar_dataset(habitacion, 'sentimiento_nlp', sentimiento)
                
                # Guardar en el listado de opiniones
                st.session_state.opiniones.append({
                    'habitacion': habitacion,
                    'opinion': opinion_texto,
                    'sentimiento': sentimiento,
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'fecha': datetime.now().strftime("%d/%m/%Y")
                })
                
                # Mostrar resultado
                st.success(f"✅ ¡Gracias por tu opinión! Sentimiento detectado: **{sentimiento}**")
                
                # Mostrar emoji según sentimiento
                if sentimiento == 'positivo':
                    st.balloons()
                elif sentimiento == 'negativo':
                    st.snow()
    
    # Mostrar últimas opiniones
    if st.session_state.opiniones:
        st.markdown("---")
        st.subheader("📋 Últimas opiniones recibidas")
        
        # Mostrar las 5 últimas opiniones
        for op in reversed(st.session_state.opiniones[-5:]):
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    if op['sentimiento'] == 'positivo':
                        st.markdown("😊 **Positivo**")
                    elif op['sentimiento'] == 'neutral':
                        st.markdown("😐 **Neutral**")
                    else:
                        st.markdown("😞 **Negativo**")
                with cols[1]:
                    st.markdown(f"**Hab {op['habitacion']}** - {op['timestamp']}")
                    st.markdown(f"_{op['opinion'][:100]}...")
                st.divider()

# =============================================================================
# VISTA INCIDENCIAS (solo si hay archivo cargado)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "⚠️ Incidencias":
    st.title("⚠️ Panel de Incidencias Operativas")
    st.caption("Falta suministros, Muy sucia, Ocupada, Otro")
    
    if st.session_state.incidencias:
        for inc in reversed(st.session_state.incidencias):
            with st.container():
                col_inc1, col_inc2 = st.columns([3, 1])
                with col_inc1:
                    st.warning(f"**{inc['timestamp']} - {inc['fecha']}**")
                    st.markdown(f"**Habitación {inc['habitacion']}** (Planta {inc['planta']})")
                    st.markdown(f"**{inc['tipo']}:** {inc['descripcion']}")
                    if 'reportado_por' in inc:
                        st.caption(f"👤 {inc['reportado_por']}")
                with col_inc2:
                    if st.button("✓ Resolver", key=f"resolver_inc_{inc['habitacion']}_{inc['timestamp']}"):
                        st.session_state.incidencias.remove(inc)
                        st.rerun()
                st.divider()
    else:
        st.info("✅ No hay incidencias operativas registradas")

# =============================================================================
# VISTA MANTENIMIENTO (solo si hay archivo cargado)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "🔧 Mantenimiento":
    st.title("🔧 Panel de Mantenimiento")
    st.caption("Averías técnicas (fontanería, electricidad, etc.)")
    
    if st.session_state.mantenimiento:
        for inc in reversed(st.session_state.mantenimiento):
            with st.container():
                col_inc1, col_inc2 = st.columns([3, 1])
                with col_inc1:
                    st.error(f"🔧 **{inc['timestamp']} - {inc['fecha']}**")
                    st.markdown(f"**Habitación {inc['habitacion']}** (Planta {inc['planta']})")
                    st.markdown(f"**{inc['tipo']}:** {inc['descripcion']}")
                    if 'reportado_por' in inc:
                        st.caption(f"👤 {inc['reportado_por']}")
                with col_inc2:
                    if st.button("✓ Reparado", key=f"resolver_mant_{inc['habitacion']}_{inc['timestamp']}"):
                        st.session_state.mantenimiento.remove(inc)
                        st.rerun()
                st.divider()
    else:
        st.info("✅ No hay averías de mantenimiento registradas")

# =============================================================================
# VISTA DATASET (solo si hay archivo cargado)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "📋 Dataset":
    st.title("📋 Dataset Enriquecido")
    
    df = st.session_state.df_pms.copy()
    
    # Añadir opiniones al dataset (por si acaso)
    if st.session_state.opiniones:
        for op in st.session_state.opiniones:
            mask = df['habitacion_id'] == op['habitacion']
            if mask.any():
                df.loc[mask, 'opinion_cliente'] = op['opinion']
                df.loc[mask, 'sentimiento_nlp'] = op['sentimiento']
    
    # Añadir clusters si están disponibles
    if st.session_state.cluster_habitaciones:
        df['cluster_kmeans'] = df['habitacion_id'].map(st.session_state.cluster_habitaciones)
    
    st.subheader("📊 Métricas del dataset")
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    with col_met1:
        st.metric("Registros", len(df))
    with col_met2:
        st.metric("Opiniones", len([o for o in st.session_state.opiniones if o]))
    with col_met3:
        st.metric("Incidencias", len(st.session_state.incidencias))
    with col_met4:
        st.metric("Mantenimiento", len(st.session_state.mantenimiento))
    
    st.subheader("📋 Datos completos")
    st.dataframe(df, use_container_width=True, height=500)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar CSV enriquecido",
        data=csv,
        file_name=f"hotel_pms_enriquecido_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# =============================================================================
# PIE DE PÁGINA (solo si hay archivo cargado)
# =============================================================================

if st.session_state.archivo_cargado:
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("🏨 **Hotel Gran Bali**")
    with col_footer2:
        st.markdown("🤖 **Sistema IA de Gestión de Limpieza**")
    with col_footer3:
        st.markdown(f"🕒 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
