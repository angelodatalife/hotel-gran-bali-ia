# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con mejoras UI y correcciones
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
            except Exception as e:
                modelos[nombre] = None
        else:
            modelos[nombre] = None
    
    return modelos

modelos = cargar_modelos()

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

def asignar_por_bloques_adyacentes(df, total_camareras=35):
    """Asigna habitaciones por BLOQUES DE PLANTAS ADYACENTES"""
    if df is None or len(df) == 0:
        return {}
    
    df_asignar = df.copy()
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
    carga_ideal_por_cam = carga_total / total_camareras
    
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
        df_bloque = df_bloque.sort_values(by=['late_checkout_pred', 'habitacion_id'], ascending=[False, True])
        
        if num_cam_bloque > 1:
            habs_por_cam = len(df_bloque) // num_cam_bloque
            resto = len(df_bloque) % num_cam_bloque
            inicio = 0
            for i in range(num_cam_bloque):
                if cam_idx > total_camareras:
                    break
                fin = inicio + habs_por_cam + (1 if i < resto else 0)
                df_cam = df_bloque.iloc[inicio:fin].copy()
                if len(df_cam) > 0:
                    asignacion[f"Camarera {cam_idx:02d}"] = df_cam
                inicio = fin
                cam_idx += 1
        else:
            if cam_idx <= total_camareras:
                asignacion[f"Camarera {cam_idx:02d}"] = df_bloque
                cam_idx += 1
    
    return asignacion

# =============================================================================
# SIDEBAR - NAVEGACIÓN (IZQUIERDA)
# =============================================================================

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
        st.markdown("")  # Espacio vacío
    with col3:
        st.markdown("")  # Espacio vacío
    
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
    
    # Dataset
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        dataset_selected = st.button("📋 Dataset", key="btn_dataset", use_container_width=True)
    with col2:
        st.markdown("")
    with col3:
        st.markdown("")
    
    # Determinar la página seleccionada
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "📊 Gerente"
    
    if gerente_selected:
        st.session_state.selected_page = "📊 Gerente"
    elif camarera_selected:
        st.session_state.selected_page = "🧹 Camarera"
    elif incidencias_selected:
        st.session_state.selected_page = "⚠️ Incidencias"
    elif mantenimiento_selected:
        st.session_state.selected_page = "🔧 Mantenimiento"
    elif dataset_selected:
        st.session_state.selected_page = "📋 Dataset"
    
    selected = st.session_state.selected_page
    
    st.markdown("---")
    
    # Cargar archivo CSV
    st.subheader("📂 Cargar PMS")
    archivo = st.file_uploader("Selecciona archivo CSV", type=['csv'], key="file_uploader_sidebar")
    
    if archivo is not None and st.session_state.df_pms is None:
        with st.spinner("Procesando archivo..."):
            df = pd.read_csv(archivo)
            
            # Aplicar ANN
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
                    st.warning(f"⚠️ No se pudo aplicar ANN")
            
            st.session_state.df_pms = df
            
            with st.spinner("Calculando asignación por bloques adyacentes..."):
                st.session_state.asignacion_por_camarera = asignar_por_bloques_adyacentes(df)
            
            st.success(f"✅ PMS cargado: {len(df)} habitaciones")
            st.rerun()
    
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
                    'asignacion_por_camarera', 'habitaciones_completadas', 'habitaciones_standby']:
            if key in st.session_state:
                if key in ['incidencias', 'mantenimiento', 'opiniones', 'habitaciones_completadas', 'habitaciones_standby']:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.rerun()

# =============================================================================
# VISTA GERENTE - CON HEATMAP Y MÉTRICAS MEJORADAS
# =============================================================================

elif selected == "📊 Gerente":
    st.title("📊 Dashboard Gerente - Hotel Gran Bali")
    
    if st.session_state.df_pms is None:
        st.warning("Carga un archivo PMS desde el menú lateral")
    else:
        df = st.session_state.df_pms
        total_habitaciones = 446
        habitaciones_limpiar = len(df)
        ocupacion = (habitaciones_limpiar / total_habitaciones) * 100
        
        # ===== MÉTRICAS PRINCIPALES =====
        st.subheader("📊 Resumen del día")
        
        col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
        with col_metric1:
            st.metric("Habitaciones", habitaciones_limpiar)
        with col_metric2:
            # Círculo de ocupación
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    background: conic-gradient(#4CAF50 0deg {ocupacion * 3.6}deg, #e0e0e0 {ocupacion * 3.6}deg 360deg);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    font-size: 18px;
                    font-weight: bold;
                ">
                    {ocupacion:.1f}%
                </div>
                <p style="margin-top: 5px;">Ocupación</p>
            </div>
            """, unsafe_allow_html=True)
        with col_metric3:
            st.metric("Camareras", "35")
        with col_metric4:
            st.metric("Stand By", len(st.session_state.habitaciones_standby))
        with col_metric5:
            if 'late_checkout_pred' in df.columns:
                checkouts = df['late_checkout_pred'].sum()
                st.metric("Late checkout", int(checkouts))
        
        st.markdown("---")
        
        # ===== HEATMAP DE HABITACIONES POR SECTOR =====
        st.subheader("🗺️ Estado de habitaciones por sector")
        
        # Definir sectores
        sectores = {
            'Bajo': {'plantas': list(range(2, 16)), 'color': '#4CAF50', 'hab_por_planta': 18},
            'Medio': {'plantas': list(range(16, 31)), 'color': '#FFC107', 'hab_por_planta': 10},
            'Alto': {'plantas': list(range(31, 53)), 'color': '#DC143C', 'hab_por_planta': 3}
        }
        
        # Ajustar plantas altas (43-52 tienen 2 habitaciones)
        for planta in range(43, 53):
            if planta in sectores['Alto']['plantas']:
                # Ya está incluida, solo necesitamos ajustar el heatmap después
                pass
        
        # Crear heatmap por sectores
        tabs = st.tabs(["Todos", "Sector Bajo", "Sector Medio", "Sector Alto"])
        
        with tabs[0]:
            st.caption("Heatmap general - Pasa el ratón para ver detalles")
            # Crear grid de todas las habitaciones
            cols_heatmap = st.columns(10)  # 10 columnas para mejor visualización
            col_idx = 0
            
            for sector_nombre, sector_info in sectores.items():
                for planta in sector_info['plantas']:
                    # Determinar número de habitaciones en esta planta
                    if sector_nombre == 'Alto' and planta >= 43:
                        num_hab_planta = 2
                    else:
                        num_hab_planta = sector_info['hab_por_planta']
                    
                    for hab_num in range(1, num_hab_planta + 1):
                        hab_id = planta * 100 + hab_num
                        
                        # Determinar estado de la habitación
                        estado = "pendiente"
                        color = "#f0f0f0"  # Gris claro por defecto
                        tooltip = f"Hab {hab_id} - Pendiente"
                        
                        # Verificar si está en completadas
                        if hab_id in st.session_state.habitaciones_completadas:
                            estado = "completada"
                            color = "#4CAF50"  # Verde
                            tooltip = f"Hab {hab_id} - Completada"
                        
                        # Verificar si está en standby
                        elif hab_id in st.session_state.habitaciones_standby:
                            # Buscar el tipo de incidencia
                            inc = next((i for i in st.session_state.incidencias if i['habitacion'] == hab_id), None)
                            if inc:
                                if inc['tipo'] == "Mantenimiento (avería)":
                                    color = "#808080"  # Gris
                                    tooltip = f"Hab {hab_id} - Mantenimiento"
                                elif inc['tipo'] == "Falta suministros":
                                    color = "#FF9800"  # Naranja
                                    tooltip = f"Hab {hab_id} - Falta suministros"
                                elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                                    color = "#f44336"  # Rojo
                                    tooltip = f"Hab {hab_id} - {inc['tipo']}"
                                else:
                                    color = "#FFC107"  # Amarillo para otros
                                    tooltip = f"Hab {hab_id} - {inc['tipo']}"
                            else:
                                color = "#FFC107"  # Amarillo por defecto
                                tooltip = f"Hab {hab_id} - En standby"
                        
                        # Verificar si está en el PMS (a limpiar hoy)
                        elif hab_id not in df['habitacion_id'].values:
                            color = "#ffffff"  # Blanco (no se limpia hoy)
                            tooltip = f"Hab {hab_id} - No se limpia hoy"
                        
                        with cols_heatmap[col_idx % 10]:
                            st.markdown(f"""
                            <div style="
                                background-color: {color};
                                width: 40px;
                                height: 40px;
                                border: 1px solid #ccc;
                                border-radius: 4px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-size: 10px;
                                margin: 2px;
                                cursor: help;
                                transition: transform 0.2s;
                                title: {tooltip};
                            " title="{tooltip}">
                                {hab_id}
                            </div>
                            """, unsafe_allow_html=True)
                        col_idx += 1
        
        with tabs[1]:
            st.caption("Sector Bajo (Plantas 2-15)")
            cols_bajo = st.columns(9)  # 9 columnas para 18 habitaciones por planta
            col_idx = 0
            for planta in range(2, 16):
                for hab_num in range(1, 19):
                    hab_id = planta * 100 + hab_num
                    
                    # Determinar color (misma lógica que arriba)
                    color = "#f0f0f0"
                    if hab_id in st.session_state.habitaciones_completadas:
                        color = "#4CAF50"
                    elif hab_id in st.session_state.habitaciones_standby:
                        inc = next((i for i in st.session_state.incidencias if i['habitacion'] == hab_id), None)
                        if inc:
                            if inc['tipo'] == "Mantenimiento (avería)":
                                color = "#808080"
                            elif inc['tipo'] == "Falta suministros":
                                color = "#FF9800"
                            elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                                color = "#f44336"
                            else:
                                color = "#FFC107"
                    elif hab_id not in df['habitacion_id'].values:
                        color = "#ffffff"
                    
                    with cols_bajo[col_idx % 9]:
                        st.markdown(f"""
                        <div style="
                            background-color: {color};
                            width: 35px;
                            height: 35px;
                            border: 1px solid #ccc;
                            border-radius: 4px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 9px;
                            margin: 2px;
                        ">
                            {hab_id}
                        </div>
                        """, unsafe_allow_html=True)
                    col_idx += 1
        
        with tabs[2]:
            st.caption("Sector Medio (Plantas 16-30)")
            cols_medio = st.columns(5)  # 5 columnas para 10 habitaciones por planta
            col_idx = 0
            for planta in range(16, 31):
                for hab_num in range(1, 11):
                    hab_id = planta * 100 + hab_num
                    
                    color = "#f0f0f0"
                    if hab_id in st.session_state.habitaciones_completadas:
                        color = "#4CAF50"
                    elif hab_id in st.session_state.habitaciones_standby:
                        inc = next((i for i in st.session_state.incidencias if i['habitacion'] == hab_id), None)
                        if inc:
                            if inc['tipo'] == "Mantenimiento (avería)":
                                color = "#808080"
                            elif inc['tipo'] == "Falta suministros":
                                color = "#FF9800"
                            elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                                color = "#f44336"
                            else:
                                color = "#FFC107"
                    elif hab_id not in df['habitacion_id'].values:
                        color = "#ffffff"
                    
                    with cols_medio[col_idx % 5]:
                        st.markdown(f"""
                        <div style="
                            background-color: {color};
                            width: 40px;
                            height: 40px;
                            border: 1px solid #ccc;
                            border-radius: 4px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 10px;
                            margin: 2px;
                        ">
                            {hab_id}
                        </div>
                        """, unsafe_allow_html=True)
                    col_idx += 1
        
        with tabs[3]:
            st.caption("Sector Alto (Plantas 31-52)")
            cols_alto = st.columns(3)  # 3 columnas
            col_idx = 0
            for planta in range(31, 53):
                num_hab = 2 if planta >= 43 else 3
                for hab_num in range(1, num_hab + 1):
                    hab_id = planta * 100 + hab_num
                    
                    color = "#f0f0f0"
                    if hab_id in st.session_state.habitaciones_completadas:
                        color = "#4CAF50"
                    elif hab_id in st.session_state.habitaciones_standby:
                        inc = next((i for i in st.session_state.incidencias if i['habitacion'] == hab_id), None)
                        if inc:
                            if inc['tipo'] == "Mantenimiento (avería)":
                                color = "#808080"
                            elif inc['tipo'] == "Falta suministros":
                                color = "#FF9800"
                            elif inc['tipo'] in ["Muy sucia", "Ocupada"]:
                                color = "#f44336"
                            else:
                                color = "#FFC107"
                    elif hab_id not in df['habitacion_id'].values:
                        color = "#ffffff"
                    
                    with cols_alto[col_idx % 3]:
                        st.markdown(f"""
                        <div style="
                            background-color: {color};
                            width: 45px;
                            height: 45px;
                            border: 1px solid #ccc;
                            border-radius: 4px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 11px;
                            margin: 2px;
                        ">
                            {hab_id}
                        </div>
                        """, unsafe_allow_html=True)
                    col_idx += 1
        
        st.markdown("---")
        
        # Leyenda de colores
        st.subheader("📖 Leyenda")
        col_leg1, col_leg2, col_leg3, col_leg4, col_leg5, col_leg6 = st.columns(6)
        with col_leg1:
            st.markdown("🟩 **Verde**: Completada")
        with col_leg2:
            st.markdown("⬜ **Blanco**: No se limpia")
        with col_leg3:
            st.markdown("🟨 **Amarillo**: Pendiente")
        with col_leg4:
            st.markdown("🟧 **Naranja**: Falta suministros")
        with col_leg5:
            st.markdown("🟥 **Rojo**: Muy sucia/Ocupada")
        with col_leg6:
            st.markdown("⬛ **Gris**: Mantenimiento")
        
        st.markdown("---")
        
        # ===== CARGA DE TRABAJO POR CAMARERA =====
        st.subheader("📊 Carga de trabajo por camarera")
        
        if st.session_state.asignacion_por_camarera:
            datos_carga = []
            for cam, df_cam in st.session_state.asignacion_por_camarera.items():
                num_hab = len(df_cam)
                # Calcular media de habitaciones por camarera
                datos_carga.append({
                    'Camarera': cam,
                    'Habitaciones': num_hab
                })
            
            df_carga = pd.DataFrame(datos_carga)
            media_hab = df_carga['Habitaciones'].mean()
            desviacion = df_carga['Habitaciones'].std()
            umbral_alerta = media_hab + desviacion  # Camareras con carga superior a la media + desviación
            
            # Crear gráfico de barras
            fig = px.bar(
                df_carga,
                x='Camarera',
                y='Habitaciones',
                title='Habitaciones asignadas por camarera',
                color='Habitaciones',
                color_continuous_scale=['#4CAF50', '#FFC107', '#f44336'],
                range_color=[0, df_carga['Habitaciones'].max()]
            )
            
            # Añadir línea de media
            fig.add_hline(
                y=media_hab,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Media: {media_hab:.1f}",
                annotation_position="top left"
            )
            
            # Marcar barras con carga alta en rojo
            for i, row in df_carga.iterrows():
                if row['Habitaciones'] > umbral_alerta:
                    fig.add_annotation(
                        x=row['Camarera'],
                        y=row['Habitaciones'],
                        text="⚠️",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red"
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar estadísticas
            col_est1, col_est2, col_est3, col_est4 = st.columns(4)
            with col_est1:
                st.metric("Media hab/cam", f"{media_hab:.1f}")
            with col_est2:
                st.metric("Mínimo", int(df_carga['Habitaciones'].min()))
            with col_est3:
                st.metric("Máximo", int(df_carga['Habitaciones'].max()))
            with col_est4:
                sobrecargadas = len(df_carga[df_carga['Habitaciones'] > umbral_alerta])
                st.metric("Sobrecargadas", sobrecargadas, delta=None, delta_color="inverse")
            
            if sobrecargadas > 0:
                st.warning(f"⚠️ {sobrecargadas} camareras tienen carga superior a la media. Considera asignar ayuda.")
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA - CORREGIDA
# =============================================================================

elif selected == "🧹 Camarera":
    st.title("🧹 App Camarera - Hotel Gran Bali")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ El gerente debe cargar el PMS primero")
    elif not st.session_state.asignacion_por_camarera:
        st.warning("⚠️ No hay asignación disponible")
    else:
        if st.session_state.camarera_actual is None:
            st.subheader("👤 Selecciona tu perfil")
            
            # Selector con dirección hacia abajo (por defecto)
            camareras = list(st.session_state.asignacion_por_camarera.keys())
            st.session_state.camarera_actual = st.selectbox(
                "Nombre:",
                camareras,
                index=None,
                placeholder="Elige tu nombre...",
                key="select_camarera_principal",
                label_visibility="collapsed"  # Oculta la label para mejor UX
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
                if st.button("🔄 Cambiar usuario", key="btn_cambiar_usuario_principal", disabled=limpieza_en_curso):
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
                            
                            df = st.session_state.df_pms
                            hab_id = hab['habitacion_id']
                            df.loc[df['habitacion_id'] == hab_id, 'tiempo_real'] = round(tiempo_real, 1)
                            st.session_state.df_pms = df
                            
                            st.session_state.habitaciones_completadas.append(hab_id)
                            
                            st.success(f"✅ Habitación {int(hab_id)} completada en {tiempo_real:.1f} minutos")
                            
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            st.session_state.reporte_expander_open = False
                            time.sleep(1)
                            st.rerun()
                    
                    # Reportar problema (solo visible durante limpieza)
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
                    
                    for hab_id in st.session_state.habitaciones_standby[:]:
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
                                    if st.button("✅ Resuelto", key=f"btn_standby_{hab_id}", disabled=limpieza_en_curso):
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
                                if 'tiempo_estimado' in row:
                                    st.markdown(f"⏱️ **{row['tiempo_estimado']} min**")
                            
                            with cols[3]:
                                # Deshabilitar botones si hay limpieza en curso
                                if limpieza_en_curso:
                                    st.button(
                                        f"⏸️ En curso", 
                                        key=f"btn_disabled_{row['habitacion_id']}_{i}",
                                        disabled=True,
                                        use_container_width=True
                                    )
                                else:
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
# VISTA INCIDENCIAS (operativas)
# =============================================================================

elif selected == "⚠️ Incidencias":
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
# VISTA MANTENIMIENTO (averías técnicas)
# =============================================================================

elif selected == "🔧 Mantenimiento":
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
# VISTA DATASET
# =============================================================================

elif selected == "📋 Dataset":
    st.title("📋 Dataset Enriquecido")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ Primero carga un archivo PMS")
    else:
        df = st.session_state.df_pms.copy()
        
        if st.session_state.opiniones:
            for op in st.session_state.opiniones:
                mask = df['habitacion_id'] == op['habitacion']
                if mask.any():
                    df.loc[mask, 'opinion_cliente'] = op['opinion']
                    df.loc[mask, 'sentimiento_nlp'] = op['sentimiento']
        
        st.subheader("📊 Métricas")
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

st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.markdown("🏨 **Hotel Gran Bali**")
with col_footer2:
    st.markdown("🤖 **Sistema IA de Gestión de Limpieza**")
with col_footer3:
    st.markdown(f"🕒 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
