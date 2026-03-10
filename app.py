# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con XGBoost integrado para predicción de tiempos
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
                st.success(f"✅ Modelo {nombre} cargado correctamente")
            except Exception as e:
                st.warning(f"⚠️ No se pudo cargar {nombre}: {str(e)}")
                modelos[nombre] = None
        else:
            st.warning(f"⚠️ Archivo no encontrado: {archivo}")
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

def predecir_tiempos_xgboost(df):
    """Aplica el modelo XGBoost para predecir tiempos de limpieza por habitación"""
    if modelos.get('xgboost') is None:
        st.warning("⚠️ Modelo XGBoost no disponible, usando tiempos del CSV")
        return df
    
    try:
        xgb_model = modelos['xgboost']['modelo']
        feature_cols = modelos['xgboost']['feature_cols']
        
        # Verificar que todas las columnas necesarias existen
        cols_disponibles = [c for c in feature_cols if c in df.columns]
        if len(cols_disponibles) == len(feature_cols):
            X_xgb = df[feature_cols].values
            # Predecir tiempos
            tiempos_pred = xgb_model.predict(X_xgb)
            df['tiempo_xgb_pred'] = np.round(tiempos_pred, 1)
            st.success(f"✅ XGBoost: tiempos predichos para {len(df)} habitaciones")
        else:
            st.warning(f"⚠️ Faltan columnas para XGBoost: {set(feature_cols) - set(cols_disponibles)}")
            df['tiempo_xgb_pred'] = df['tiempo_estimado'] if 'tiempo_estimado' in df.columns else 25
    except Exception as e:
        st.warning(f"⚠️ Error al aplicar XGBoost: {str(e)}")
        df['tiempo_xgb_pred'] = df['tiempo_estimado'] if 'tiempo_estimado' in df.columns else 25
    
    return df

def asignar_por_bloques_adyacentes(df, num_camareras=TOTAL_CAMARERAS):
    """Asigna habitaciones por BLOQUES DE PLANTAS ADYACENTES usando tiempos de XGBoost"""
    if df is None or len(df) == 0:
        return {}
    
    df_asignar = df.copy()
    plantas_totales = sorted(df_asignar['planta'].unique())
    
    # Usar tiempos de XGBoost si están disponibles, sino los del CSV
    if 'tiempo_xgb_pred' in df_asignar.columns:
        tiempo_col = 'tiempo_xgb_pred'
    else:
        tiempo_col = 'tiempo_estimado' if 'tiempo_estimado' in df_asignar.columns else None
    
    # 1. Calcular carga por planta (basada en tiempo, no en número de habitaciones)
    carga_por_planta = {}
    for planta in plantas_totales:
        df_planta = df_asignar[df_asignar['planta'] == planta]
        if tiempo_col and tiempo_col in df_planta.columns:
            carga_por_planta[planta] = df_planta[tiempo_col].sum()
        else:
            # Si no hay tiempos, estimar 25 min por habitación
            carga_por_planta[planta] = len(df_planta) * 25
    
    # 2. Calcular carga total y carga ideal por camarera
    carga_total = sum(carga_por_planta.values())
    carga_ideal_por_cam = carga_total / num_camareras
    
    # 3. Aplicar ANN para priorizar urgentes
    if modelos.get('ann') is not None:
        try:
            ann_model = modelos['ann']['modelo']
            scaler_ann = modelos['ann']['scaler']
            feature_cols = modelos['ann']['feature_cols']
            
            cols_disponibles = [c for c in feature_cols if c in df_asignar.columns]
            if len(cols_disponibles) == len(feature_cols):
                X_ann = df_asignar[feature_cols].values
                X_ann_scaled = scaler_ann.transform(X_ann)
                
                prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                df_asignar['prob_late'] = prob_late
                df_asignar['late_checkout_pred'] = (prob_late > 0.5).astype(int)
            else:
                df_asignar['late_checkout_pred'] = 0
        except:
            df_asignar['late_checkout_pred'] = 0
    else:
        df_asignar['late_checkout_pred'] = 0
    
    # 4. Crear BLOQUES DE PLANTAS ADYACENTES basados en carga
    bloques = []
    bloque_actual = []
    carga_acumulada = 0
    
    for planta in plantas_totales:
        bloque_actual.append(planta)
        carga_acumulada += carga_por_planta[planta]
        
        # Cerrar bloque cuando alcanza ~80% de la carga ideal
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
    
    # 5. Asignar camareras a bloques (proporcional a la carga)
    asignacion = {}
    cam_idx = 1
    
    for bloque in bloques:
        plantas_bloque = bloque['plantas']
        carga_bloque = bloque['carga']
        
        # Calcular cuántas camareras necesita este bloque
        num_cam_bloque = max(1, round(carga_bloque / carga_ideal_por_cam))
        df_bloque = df_asignar[df_asignar['planta'].isin(plantas_bloque)].copy()
        df_bloque = df_bloque.sort_values(by=['late_checkout_pred', 'habitacion_id'], ascending=[False, True])
        
        if num_cam_bloque > 1 and len(df_bloque) >= num_cam_bloque:
            # Dividir equitativamente por carga (tiempo), no por número de habitaciones
            # Ordenar por tiempo para balancear
            if tiempo_col in df_bloque.columns:
                df_bloque = df_bloque.sort_values(by=[tiempo_col], ascending=False)
            
            # Calcular carga por camarera en este bloque
            carga_por_cam_bloque = carga_bloque / num_cam_bloque
            
            inicio = 0
            for i in range(num_cam_bloque):
                if cam_idx > num_camareras:
                    break
                
                # Acumular habitaciones hasta alcanzar la carga deseada
                carga_actual = 0
                fin = inicio
                for j in range(inicio, len(df_bloque)):
                    if tiempo_col in df_bloque.columns:
                        carga_actual += df_bloque.iloc[j][tiempo_col]
                    else:
                        carga_actual += 25
                    fin = j + 1
                    if carga_actual >= carga_por_cam_bloque * 0.8 or j == len(df_bloque) - 1:
                        break
                
                df_cam = df_bloque.iloc[inicio:fin].copy()
                if len(df_cam) > 0:
                    asignacion[f"Camarera {cam_idx:02d}"] = df_cam
                inicio = fin
                cam_idx += 1
        else:
            # Una sola camarera para este bloque
            if cam_idx <= num_camareras:
                asignacion[f"Camarera {cam_idx:02d}"] = df_bloque
                cam_idx += 1
    
    # 6. Distribuir camareras restantes si no se llegó al total
    if cam_idx <= num_camareras:
        # Recolectar todas las habitaciones asignadas
        todas_habitaciones = []
        for cam, df_cam in asignacion.items():
            todas_habitaciones.extend(df_cam['habitacion_id'].tolist())
        
        # Identificar habitaciones no asignadas (si las hay)
        todas_hab_df = set(df_asignar['habitacion_id'].tolist())
        asignadas = set(todas_habitaciones)
        no_asignadas = list(todas_hab_df - asignadas)
        
        if no_asignadas:
            # Hay habitaciones sin asignar, repartirlas entre las camareras restantes
            df_no_asignadas = df_asignar[df_asignar['habitacion_id'].isin(no_asignadas)]
            
            # Ordenar por tiempo para balancear
            if tiempo_col in df_no_asignadas.columns:
                df_no_asignadas = df_no_asignadas.sort_values(by=[tiempo_col], ascending=False)
            
            habs_por_cam_restantes = len(df_no_asignadas) // (num_camareras - cam_idx + 1)
            
            inicio = 0
            for i in range(cam_idx, num_camareras + 1):
                fin = inicio + habs_por_cam_restantes
                if i == num_camareras:
                    fin = len(df_no_asignadas)  # La última toma todas las restantes
                df_cam = df_no_asignadas.iloc[inicio:fin].copy() if inicio < len(df_no_asignadas) else pd.DataFrame()
                asignacion[f"Camarera {i:02d}"] = df_cam
                inicio = fin
    
    # 7. Asegurar que tenemos exactamente 35 camareras en orden
    asignacion_final = {}
    for i in range(1, num_camareras + 1):
        cam_name = f"Camarera {i:02d}"
        if cam_name in asignacion:
            asignacion_final[cam_name] = asignacion[cam_name]
        else:
            # Asignar DataFrame vacío para camareras sin habitaciones
            asignacion_final[cam_name] = pd.DataFrame()
    
    return asignacion_final

def actualizar_dataset(hab_id, campo, valor):
    """Actualiza una columna específica en el dataset para una habitación"""
    if st.session_state.df_pms is not None:
        df = st.session_state.df_pms
        if hab_id in df['habitacion_id'].values:
            df.loc[df['habitacion_id'] == hab_id, campo] = valor
            st.session_state.df_pms = df

def procesar_archivo(archivo):
    """Procesa el archivo cargado y aplica todos los modelos"""
    with st.spinner("Procesando archivo..."):
        df = pd.read_csv(archivo)
        
        # Asegurar que existen las columnas necesarias
        columnas_necesarias = ['tiempo_real', 'incidencia_camarera', 'opinion_cliente', 'sentimiento_nlp']
        for col in columnas_necesarias:
            if col not in df.columns:
                df[col] = None
        
        # 1. APLICAR XGBOOST para predecir tiempos
        with st.spinner("Aplicando XGBoost para predicción de tiempos..."):
            df = predecir_tiempos_xgboost(df)
        
        # 2. APLICAR ANN para predecir late checkout
        if modelos.get('ann') is not None:
            try:
                ann_model = modelos['ann']['modelo']
                scaler_ann = modelos['ann']['scaler']
                feature_cols = modelos['ann']['feature_cols']
                
                cols_disponibles = [c for c in feature_cols if c in df.columns]
                if len(cols_disponibles) == len(feature_cols):
                    X_ann = df[feature_cols].values
                    X_ann_scaled = scaler_ann.transform(X_ann)
                    
                    prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                    df['prob_late'] = prob_late
                    df['late_checkout_pred'] = (prob_late > 0.5).astype(int)
            except Exception as e:
                st.warning(f"⚠️ No se pudo aplicar ANN: {str(e)}")
        
        st.session_state.df_pms = df
        
        # 3. APLICAR ASIGNACIÓN con K-Means y XGBoost
        with st.spinner("Calculando asignación por bloques adyacentes (usando XGBoost y K-Means)..."):
            st.session_state.asignacion_por_camarera = asignar_por_bloques_adyacentes(df, st.session_state.num_camareras)
        
        st.session_state.archivo_cargado = True
        st.session_state.selected_page = "📊 Gerente"
        
        # Mostrar resumen de modelos aplicados
        st.success(f"✅ PMS cargado: {len(df)} habitaciones")
        st.success(f"✅ XGBoost: tiempos predichos")
        st.success(f"✅ ANN: prioridades calculadas")
        st.success(f"✅ K-Means: asignación por bloques")
        time.sleep(2)
        st.rerun()

# =============================================================================
# PANTALLA DE INICIO (antes de cargar archivo)
# =============================================================================

def mostrar_pantalla_inicio():
    """Muestra la pantalla de inicio centralizada"""
    
    st.markdown(
        """
        <h2 style='text-align: center; color: #1E88E5; margin-top: 50px; margin-bottom: 30px;'>
            🏨 Hotel Gran Bali - Sistema de Gestión IA de Limpieza
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <h3 style='text-align: center; color: #333; margin-bottom: 20px;'>
                Cargar PMS aquí ⬇️
            </h3>
            """,
            unsafe_allow_html=True
        )
        
        archivo = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type=['csv'],
            key="file_uploader_inicio",
            label_visibility="collapsed"
        )
        
        if archivo is not None:
            procesar_archivo(archivo)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Sección de modelos cargados
        st.markdown(
            """
            <h4 style='text-align: center; color: #666; margin-bottom: 20px;'>
                🤖 Modelos cargados
            </h4>
            """,
            unsafe_allow_html=True
        )
        
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
# SIDEBAR - NAVEGACIÓN
# =============================================================================

def mostrar_sidebar():
    """Muestra el sidebar con la navegación"""
    with st.sidebar:
        st.title("🏨 Hotel Gran Bali")
        st.markdown("---")
        
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
                        'asignacion_por_camarera', 'habitaciones_completadas', 'habitaciones_standby', 'archivo_cargado']:
                if key in st.session_state:
                    if key in ['incidencias', 'mantenimiento', 'opiniones', 'habitaciones_completadas', 'habitaciones_standby']:
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            st.session_state.archivo_cargado = False
            st.rerun()

# =============================================================================
# LÓGICA PRINCIPAL DE NAVEGACIÓN
# =============================================================================

if not st.session_state.archivo_cargado or st.session_state.df_pms is None:
    mostrar_pantalla_inicio()
else:
    mostrar_sidebar()
    selected = st.session_state.selected_page

# =============================================================================
# VISTA GERENTE
# =============================================================================

if st.session_state.archivo_cargado and selected == "📊 Gerente":
    
    df = st.session_state.df_pms
    
    tab_dashboard, tab_estado, tab_carga = st.tabs(["📊 Dashboard Gerente", "🗺️ Estado de Habitaciones", "📊 Carga de trabajo por camarera"])
    
    with tab_dashboard:
        st.title("📊 Dashboard Gerente - Hotel Gran Bali")
        
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
        
        st.subheader("📈 Resumen rápido")
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.metric("Checkouts estimados", int(df['late_checkout_pred'].sum()) if 'late_checkout_pred' in df.columns else 0)
        with col_res2:
            st.metric("Repasos", len(df) - (int(df['late_checkout_pred'].sum()) if 'late_checkout_pred' in df.columns else 0))
        with col_res3:
            st.metric("Incidencias", len(st.session_state.incidencias))
        with col_res4:
            st.metric("Mantenimiento", len(st.session_state.mantenimiento))
        
        # Mostrar estadísticas de XGBoost
        if 'tiempo_xgb_pred' in df.columns:
            st.info(f"⏱️ Tiempo total estimado por XGBoost: {df['tiempo_xgb_pred'].sum():.0f} minutos")
    
    with tab_estado:
        st.title("🗺️ Estado de Habitaciones")
        
        sectores = {
            'Bajo (Pl 2-15)': list(range(2, 16)),
            'Medio (Pl 16-30)': list(range(16, 31)),
            'Alto (Pl 31-52)': list(range(31, 53))
        }
        
        def get_room_color(hab_id):
            if hab_id in st.session_state.habitaciones_completadas:
                for inc in st.session_state.incidencias:
                    if inc['habitacion'] == hab_id:
                        if inc['tipo'] == "Falta suministros":
                            return "#FFA500"
                        elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                            return "#FF4444"
                for mant in st.session_state.mantenimiento:
                    if mant['habitacion'] == hab_id:
                        return "#808080"
                return "#4CAF50"
            elif hab_id in st.session_state.habitaciones_standby:
                for inc in st.session_state.incidencias:
                    if inc['habitacion'] == hab_id:
                        if inc['tipo'] == "Falta suministros":
                            return "#FFA500"
                        elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                            return "#FF4444"
                return "#FFA500"
            return "#FFFFFF"
        
        subtab1, subtab2, subtab3 = st.tabs(["🔵 Sector Bajo", "🟡 Sector Medio", "🔴 Sector Alto"])
        
        for subtab, (sector_nombre, plantas) in zip([subtab1, subtab2, subtab3], sectores.items()):
            with subtab:
                df_sector = df[df['planta'].isin(plantas)]
                
                if len(df_sector) == 0:
                    st.info(f"No hay habitaciones en {sector_nombre}")
                    continue
                
                df_sector = df_sector.sort_values('habitacion_id')
                cols_por_fila = 6
                habitaciones = df_sector['habitacion_id'].tolist()
                
                for i in range(0, len(habitaciones), cols_por_fila):
                    cols = st.columns(cols_por_fila)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(habitaciones):
                            hab_id = habitaciones[idx]
                            color = get_room_color(hab_id)
                            
                            tooltip = f"Hab {hab_id}"
                            row = df_sector[df_sector['habitacion_id'] == hab_id].iloc[0]
                            if 'tiempo_xgb_pred' in row:
                                tooltip += f"\nTiempo XGB: {row['tiempo_xgb_pred']} min"
                            elif 'tiempo_estimado' in row:
                                tooltip += f"\nTiempo est: {row['tiempo_estimado']} min"
                            
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
                            col.markdown("")
        
        st.markdown("---")
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
    
    with tab_carga:
        st.title("📊 Carga de trabajo por camarera (basada en XGBoost)")
        
        if st.session_state.asignacion_por_camarera:
            datos_todas = []
            for i in range(1, st.session_state.num_camareras + 1):
                cam = f"Camarera {i:02d}"
                if cam in st.session_state.asignacion_por_camarera:
                    df_cam = st.session_state.asignacion_por_camarera[cam]
                    num_hab = len(df_cam)
                    
                    # Calcular tiempo total con XGBoost si está disponible
                    if num_hab > 0 and 'tiempo_xgb_pred' in df_cam.columns:
                        tiempo_total = df_cam['tiempo_xgb_pred'].sum()
                    elif num_hab > 0 and 'tiempo_estimado' in df_cam.columns:
                        tiempo_total = df_cam['tiempo_estimado'].sum()
                    else:
                        tiempo_total = num_hab * 25
                    
                    datos_todas.append({
                        'Camarera': cam,
                        'Habitaciones': num_hab,
                        'Tiempo total (min)': round(tiempo_total, 1)
                    })
            
            datos_todas.sort(key=lambda x: int(x['Camarera'].split()[1]))
            df_carga_total = pd.DataFrame(datos_todas)
            
            col_est1, col_est2, col_est3, col_est4 = st.columns(4)
            with col_est1:
                st.metric("Total habitaciones", df_carga_total['Habitaciones'].sum())
            with col_est2:
                st.metric("Media hab/cam", f"{df_carga_total['Habitaciones'].mean():.1f}")
            with col_est3:
                st.metric("Tiempo total (min)", f"{df_carga_total['Tiempo total (min)'].sum():.0f}")
            with col_est4:
                st.metric("Media tiempo/cam", f"{df_carga_total['Tiempo total (min)'].mean():.0f} min")
            
            fig = px.bar(
                df_carga_total,
                x='Camarera',
                y='Tiempo total (min)',
                title='Distribución de tiempo por camarera (predicciones XGBoost)',
                color_discrete_sequence=['#1E88E5']
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Detalle por camarera")
            st.dataframe(df_carga_total, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de asignación disponibles")

# =============================================================================
# VISTA CAMARERA (se mantiene igual que antes)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "🧹 Camarera":
    st.title("🧹 App Camarera - Hotel Gran Bali")
    
    if not st.session_state.asignacion_por_camarera:
        st.warning("⚠️ No hay asignación disponible")
    else:
        if st.session_state.camarera_actual is None:
            st.subheader("👤 Selecciona tu perfil")
            
            camareras_con_hab = []
            for i in range(1, st.session_state.num_camareras + 1):
                cam = f"Camarera {i:02d}"
                if cam in st.session_state.asignacion_por_camarera:
                    df_cam = st.session_state.asignacion_por_camarera[cam]
                    if len(df_cam) > 0:
                        camareras_con_hab.append(cam)
            
            st.session_state.camarera_actual = st.selectbox(
                "Nombre:",
                camareras_con_hab,
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
            limpieza_en_curso = st.session_state.cronometro_activo and st.session_state.habitacion_actual is not None
            
            df_asignadas = st.session_state.asignacion_por_camarera.get(
                st.session_state.camarera_actual, 
                pd.DataFrame()
            )
            
            total_asignadas = len(df_asignadas)
            completadas = len(st.session_state.habitaciones_completadas)
            pendientes = total_asignadas - completadas - len(st.session_state.habitaciones_standby)
            
            col_info1, col_info2, col_info3, col_info4 = st.columns([2, 2, 2, 1])
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                if len(df_asignadas) > 0:
                    plantas_unicas = sorted(df_asignadas['planta'].unique())
                    st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)}")
            with col_info3:
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
            
            if total_asignadas > 0:
                progreso_total = completadas / total_asignadas
                st.progress(progreso_total, text=f"**{completadas}/{total_asignadas}** habitaciones completadas")
            
            if limpieza_en_curso:
                with st.container():
                    st.subheader("⏱️ Limpieza en curso")
                    
                    hab = st.session_state.habitacion_actual
                    
                    col_crono1, col_crono2, col_crono3 = st.columns(3)
                    
                    with col_crono1:
                        st.markdown(f"**Habitación:** {int(hab['habitacion_id'])}")
                        st.markdown(f"**Planta:** {int(hab['planta'])}")
                    
                    with col_crono2:
                        tiempo_transcurrido = (datetime.now() - st.session_state.tiempo_inicio).seconds
                        minutos = tiempo_transcurrido // 60
                        segundos = tiempo_transcurrido % 60
                        st.markdown(f"**Tiempo:** {minutos}:{segundos:02d}")
                        
                        # Mostrar tiempo estimado por XGBoost si está disponible
                        if 'tiempo_xgb_pred' in hab:
                            st.caption(f"⏱️ Estimado XGB: {hab['tiempo_xgb_pred']} min")
                        elif 'tiempo_estimado' in hab:
                            st.caption(f"⏱️ Estimado: {hab['tiempo_estimado']} min")
                        
                        if 'tiempo_estimado' in hab:
                            progreso = min(tiempo_transcurrido / (hab['tiempo_estimado'] * 60), 1.0)
                            st.progress(progreso)
                    
                    with col_crono3:
                        if st.button("✅ Finalizar limpieza", type="primary", use_container_width=True, key="btn_finalizar_principal"):
                            tiempo_real = (datetime.now() - st.session_state.tiempo_inicio).seconds / 60
                            
                            hab_id = hab['habitacion_id']
                            actualizar_dataset(hab_id, 'tiempo_real', round(tiempo_real, 1))
                            
                            st.session_state.habitaciones_completadas.append(hab_id)
                            
                            st.success(f"✅ Habitación {int(hab_id)} completada en {tiempo_real:.1f} minutos")
                            
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            st.session_state.reporte_expander_open = False
                            time.sleep(1)
                            st.rerun()
                    
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
                                
                                actualizar_dataset(hab_id, 'incidencia_camarera', f"{tipo_reporte}: {desc_reporte}")
                                
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
                                else:
                                    st.session_state.habitaciones_standby.append(hab_id)
                                    
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
            
            if st.session_state.habitaciones_standby:
                with st.container():
                    st.subheader("⏸️ Stand By")
                    st.caption("Habitaciones pendientes por problemas")
                    
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
                                    inc = next((i for i in st.session_state.incidencias if i['habitacion'] == hab_id), None)
                                    if inc:
                                        st.markdown(f"**{inc['tipo']}**")
                                with cols[2]:
                                    if 'tiempo_xgb_pred' in row:
                                        st.markdown(f"⏱️ {row['tiempo_xgb_pred']} min")
                                    elif 'tiempo_estimado' in row:
                                        st.markdown(f"⏱️ {row['tiempo_estimado']} min")
                                with cols[3]:
                                    if st.button("✅ Resuelto", key=f"btn_standby_{hab_id}"):
                                        st.session_state.habitaciones_completadas.append(hab_id)
                                        st.session_state.habitaciones_standby.remove(hab_id)
                                        st.rerun()
                                st.divider()
            
            if total_asignadas > 0:
                df_pendientes = df_asignadas[
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_completadas) &
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_standby)
                ].copy()
                
                if 'late_checkout_pred' in df_pendientes.columns:
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
                            
                            with cols[1]:
                                if 'late_checkout_pred' in row and row['late_checkout_pred'] == 1:
                                    st.markdown("🏃 **Late**")
                                else:
                                    st.markdown("🛏️ **Normal**")
                            
                            with cols[2]:
                                if 'tiempo_xgb_pred' in row:
                                    st.markdown(f"⏱️ **{row['tiempo_xgb_pred']} min**")
                                elif 'tiempo_estimado' in row:
                                    st.markdown(f"⏱️ **{row['tiempo_estimado']} min**")
                            
                            with cols[3]:
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
                            if 'tiempo_xgb_pred' in row:
                                st.markdown(f"~~{row['tiempo_xgb_pred']} min~~")
                            elif 'tiempo_estimado' in row:
                                st.markdown(f"~~{row['tiempo_estimado']} min~~")
                        
                        with cols[3]:
                            if 'tiempo_real' in row and pd.notna(row['tiempo_real']):
                                st.markdown(f"✅ Real: {row['tiempo_real']} min")
                            else:
                                st.markdown("✅ Listo")
                        
                        st.divider()

# =============================================================================
# VISTA CLIENTE (se mantiene igual)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "👤 Cliente":
    st.title("👤 Opinión de Clientes")
    st.caption("Comparte tu experiencia para ayudarnos a mejorar")
    
    df = st.session_state.df_pms
    
    with st.form("formulario_opinion_cliente"):
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        st.markdown("#### ¿En qué podríamos mejorar?")
        opinion_texto = st.text_area(
            "Escribe tu opinión aquí:",
            placeholder="Ej: La habitación estaba muy limpia, pero el aire acondicionado hacía ruido...",
            height=150
        )
        
        submitted = st.form_submit_button("📤 Enviar opinión", use_container_width=True, type="primary")
        
        if submitted:
            if not habitacion:
                st.error("❌ Por favor, selecciona tu número de habitación")
            elif not opinion_texto:
                st.error("❌ Por favor, escribe tu opinión")
            else:
                sentimiento = procesar_opinion(opinion_texto)
                
                actualizar_dataset(habitacion, 'opinion_cliente', opinion_texto)
                actualizar_dataset(habitacion, 'sentimiento_nlp', sentimiento)
                
                st.session_state.opiniones.append({
                    'habitacion': habitacion,
                    'opinion': opinion_texto,
                    'sentimiento': sentimiento,
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'fecha': datetime.now().strftime("%d/%m/%Y")
                })
                
                st.success(f"✅ ¡Gracias por tu opinión! Sentimiento detectado: **{sentimiento}**")
                
                if sentimiento == 'positivo':
                    st.balloons()
                elif sentimiento == 'negativo':
                    st.snow()
    
    if st.session_state.opiniones:
        st.markdown("---")
        st.subheader("📋 Últimas opiniones recibidas")
        
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
# VISTA INCIDENCIAS (se mantiene igual)
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
# VISTA MANTENIMIENTO (se mantiene igual)
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
# VISTA DATASET (se mantiene igual)
# =============================================================================

elif st.session_state.archivo_cargado and selected == "📋 Dataset":
    st.title("📋 Dataset Enriquecido")
    
    df = st.session_state.df_pms.copy()
    
    if st.session_state.opiniones:
        for op in st.session_state.opiniones:
            mask = df['habitacion_id'] == op['habitacion']
            if mask.any():
                df.loc[mask, 'opinion_cliente'] = op['opinion']
                df.loc[mask, 'sentimiento_nlp'] = op['sentimiento']
    
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
# PIE DE PÁGINA
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
