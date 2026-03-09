# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con Stand By y contadores en menú
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
                st.warning(f"⚠️ No se pudo cargar {nombre}")
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
    st.session_state.incidencias = []  # Problemas operativos (Falta suministro, Muy sucia, Ocupada, Otro)
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
if 'standby_por_camarera' not in st.session_state:
    st.session_state.standby_por_camarera = {}  # Diccionario: {camarera: [lista de habitaciones en stand by]}

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
    """
    Asigna habitaciones por BLOQUES DE PLANTAS ADYACENTES
    - Divide las plantas en bloques consecutivos
    - Usa K-Means para priorizar urgentes dentro del bloque
    - Equilibra carga por tiempo estimado
    """
    if df is None or len(df) == 0:
        return {}
    
    df_asignar = df.copy()
    plantas_totales = sorted(df_asignar['planta'].unique())
    
    # 1. Calcular carga por planta (tiempo estimado total)
    carga_por_planta = {}
    for planta in plantas_totales:
        df_planta = df_asignar[df_asignar['planta'] == planta]
        if 'tiempo_estimado' in df_planta.columns:
            carga_por_planta[planta] = df_planta['tiempo_estimado'].sum()
        else:
            carga_por_planta[planta] = len(df_planta) * 25  # estimado
    
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
        
        # Si alcanzamos o superamos la carga ideal, cerramos el bloque
        if carga_acumulada >= carga_ideal_por_cam * 0.8:  # 80% de la carga ideal
            bloques.append({
                'plantas': bloque_actual.copy(),
                'carga': carga_acumulada
            })
            bloque_actual = []
            carga_acumulada = 0
    
    # Añadir el último bloque si quedó algo
    if bloque_actual:
        bloques.append({
            'plantas': bloque_actual,
            'carga': carga_acumulada
        })
    
    # 5. Asignar camareras a bloques (1-2 camareras por bloque según carga)
    asignacion = {}
    cam_idx = 1
    
    for bloque in bloques:
        plantas_bloque = bloque['plantas']
        carga_bloque = bloque['carga']
        
        # Calcular cuántas camareras necesita este bloque
        num_cam_bloque = max(1, round(carga_bloque / carga_ideal_por_cam))
        
        # Obtener habitaciones de estas plantas
        df_bloque = df_asignar[df_asignar['planta'].isin(plantas_bloque)].copy()
        
        # Ordenar por prioridad
        df_bloque = df_bloque.sort_values(
            by=['late_checkout_pred', 'habitacion_id'], 
            ascending=[False, True]
        )
        
        # Dividir habitaciones entre las camareras del bloque
        if num_cam_bloque > 1:
            # Dividir equitativamente
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
            # Una sola camarera para todo el bloque
            if cam_idx <= total_camareras:
                asignacion[f"Camarera {cam_idx:02d}"] = df_bloque
                cam_idx += 1
    
    # 6. Distribuir camareras restantes si no se llegó a 35
    if cam_idx <= total_camareras:
        # Buscar bloques con más carga y subdividir
        bloques_ordenados = sorted(bloques, key=lambda x: x['carga'], reverse=True)
        
        for bloque in bloques_ordenados:
            if cam_idx > total_camareras:
                break
            
            plantas_bloque = bloque['plantas']
            df_bloque = df_asignar[df_asignar['planta'].isin(plantas_bloque)].copy()
            
            # Verificar cuántas camareras ya tienen este bloque
            cam_en_bloque = [cam for cam, df_cam in asignacion.items() 
                            if set(df_cam['planta'].unique()) & set(plantas_bloque)]
            
            if len(cam_en_bloque) < 3:  # Máximo 3 camareras por bloque
                # Dividir el bloque en dos
                mitad = len(df_bloque) // 2
                df_cam1 = df_bloque.iloc[:mitad].copy()
                df_cam2 = df_bloque.iloc[mitad:].copy()
                
                if len(df_cam1) > 0 and cam_idx <= total_camareras:
                    asignacion[f"Camarera {cam_idx:02d}"] = df_cam1
                    cam_idx += 1
                if len(df_cam2) > 0 and cam_idx <= total_camareras:
                    asignacion[f"Camarera {cam_idx:02d}"] = df_cam2
                    cam_idx += 1
    
    return asignacion

# =============================================================================
# SIDEBAR - NAVEGACIÓN CON CONTADORES
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hotel.png", width=80)
    st.title("🏨 Hotel Gran Bali")
    st.markdown("---")
    
    # Contadores para el menú
    incidencias_count = len(st.session_state.incidencias)
    mantenimiento_count = len(st.session_state.mantenimiento)
    
    # Opciones del menú con contadores
    menu_options = {
        "📊 Gerente": "📊 Gerente",
        "🧹 Camarera": "🧹 Camarera",
        f"⚠️ Incidencias {'(' + str(incidencias_count) + ')' if incidencias_count > 0 else ''}": "⚠️ Incidencias",
        f"🔧 Mantenimiento {'(' + str(mantenimiento_count) + ')' if mantenimiento_count > 0 else ''}": "🔧 Mantenimiento",
        "📋 Dataset": "📋 Dataset"
    }
    
    selected_display = st.radio(
        "**Menú Principal**",
        list(menu_options.keys()),
        format_func=lambda x: x
    )
    
    # Obtener el valor real de la opción seleccionada
    pagina = menu_options[selected_display]
    
    st.markdown("---")
    
    with st.expander("ℹ️ Información del Sistema"):
        st.markdown("""
        **Modelos cargados:**
        - ✅ ANN (clasificación)
        - ✅ XGBoost (tiempos)
        - ✅ K-Means (clusters)
        - ✅ NLP (sentimiento)
        
        **Estado:** Activo
        """)
    
    if st.button("🔄 Reiniciar Simulación", use_container_width=True):
        for key in ['df_pms', 'incidencias', 'mantenimiento', 'opiniones', 'camarera_actual', 
                    'cronometro_activo', 'tiempo_inicio', 'habitacion_actual',
                    'asignacion_por_camarera', 'habitaciones_completadas', 'standby_por_camarera']:
            if key in st.session_state:
                if key in ['incidencias', 'mantenimiento', 'opiniones', 'habitaciones_completadas', 'standby_por_camarera']:
                    st.session_state[key] = [] if key != 'standby_por_camarera' else {}
                else:
                    st.session_state[key] = None
        st.rerun()

# =============================================================================
# VISTA GERENTE (sin cambios)
# =============================================================================

if pagina == "📊 Gerente":
    st.title("📊 Dashboard Gerente - Hotel Gran Bali")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📂 Cargar PMS del día")
        archivo = st.file_uploader("Selecciona archivo CSV", type=['csv'], key="file_uploader_gerente")
        
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
                
                # Hacer asignación por bloques adyacentes
                with st.spinner("Calculando asignación por bloques adyacentes..."):
                    st.session_state.asignacion_por_camarera = asignar_por_bloques_adyacentes(df)
                
                st.success(f"✅ PMS cargado: {len(df)} habitaciones | 35 camareras asignadas")
                st.rerun()
    
    with col2:
        if st.session_state.df_pms is not None:
            df = st.session_state.df_pms
            st.subheader("📊 Resumen del día")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Habitaciones", len(df))
            with col_metric2:
                if 'late_checkout_pred' in df.columns:
                    checkouts = df['late_checkout_pred'].sum()
                    st.metric("Late checkout", int(checkouts))
                else:
                    checkouts = len(df[df['clase_checkout'] == 'Salida']) if 'clase_checkout' in df.columns else 0
                    st.metric("Checkouts", checkouts)
            with col_metric3:
                st.metric("Camareras", "35")
    
    if st.session_state.df_pms is not None:
        df = st.session_state.df_pms
        st.markdown("---")
        
        # ===== DISTRIBUCIÓN POR SECTORES =====
        st.subheader("🏢 Distribución por Sectores")
        
        # Definir sectores
        sectores = {
            'Bajo (Pl 2-15)': {'plantas': list(range(2, 16)), 'camareras_base': 19},
            'Medio (Pl 16-30)': {'plantas': list(range(16, 31)), 'camareras_base': 11},
            'Alto (Pl 31-52)': {'plantas': list(range(31, 53)), 'camareras_base': 5}
        }
        
        # Calcular datos por sector
        datos_sectores = []
        for sector, info in sectores.items():
            df_sector = df[df['planta'].isin(info['plantas'])]
            num_hab = len(df_sector)
            
            # Calcular camareras asignadas realmente (desde la asignación)
            cam_asignadas = 0
            if st.session_state.asignacion_por_camarera:
                for cam, df_cam in st.session_state.asignacion_por_camarera.items():
                    if any(p in info['plantas'] for p in df_cam['planta'].unique()):
                        cam_asignadas += 1
            
            # Usar base si no hay asignación
            if cam_asignadas == 0:
                cam_asignadas = info['camareras_base']
            
            # Calcular tiempo total del sector
            tiempo_total = df_sector['tiempo_estimado'].sum() if 'tiempo_estimado' in df_sector.columns else num_hab * 25
            
            datos_sectores.append({
                'Sector': sector,
                'Habitaciones': num_hab,
                'Camareras': cam_asignadas,
                'Hab/Cam': round(num_hab / cam_asignadas, 1) if cam_asignadas > 0 else 0,
                'Tiempo total (min)': round(tiempo_total, 1),
                'Tiempo/cam (min)': round(tiempo_total / cam_asignadas, 1) if cam_asignadas > 0 else 0
            })
        
        df_sectores = pd.DataFrame(datos_sectores)
        
        # Mostrar métricas por sector en columnas
        col_sect1, col_sect2, col_sect3 = st.columns(3)
        
        with col_sect1:
            st.markdown("### 🔵 Sector Bajo")
            st.metric("Habitaciones", df_sectores.iloc[0]['Habitaciones'])
            st.metric("Camareras", df_sectores.iloc[0]['Camareras'])
            st.metric("Hab/Cam", df_sectores.iloc[0]['Hab/Cam'])
            st.metric("Tiempo total", f"{df_sectores.iloc[0]['Tiempo total (min)']} min")
        
        with col_sect2:
            st.markdown("### 🟡 Sector Medio")
            st.metric("Habitaciones", df_sectores.iloc[1]['Habitaciones'])
            st.metric("Camareras", df_sectores.iloc[1]['Camareras'])
            st.metric("Hab/Cam", df_sectores.iloc[1]['Hab/Cam'])
            st.metric("Tiempo total", f"{df_sectores.iloc[1]['Tiempo total (min)']} min")
        
        with col_sect3:
            st.markdown("### 🔴 Sector Alto")
            st.metric("Habitaciones", df_sectores.iloc[2]['Habitaciones'])
            st.metric("Camareras", df_sectores.iloc[2]['Camareras'])
            st.metric("Hab/Cam", df_sectores.iloc[2]['Hab/Cam'])
            st.metric("Tiempo total", f"{df_sectores.iloc[2]['Tiempo total (min)']} min")
        
        st.markdown("---")
        
        # ===== GRÁFICOS DE SECTORES =====
        col_graf_sect1, col_graf_sect2 = st.columns(2)
        
        with col_graf_sect1:
            # Gráfico de barras: Habitaciones vs Camareras
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Habitaciones',
                x=df_sectores['Sector'],
                y=df_sectores['Habitaciones'],
                marker_color=['#1E88E5', '#FFC107', '#DC143C']
            ))
            fig.add_trace(go.Bar(
                name='Camareras',
                x=df_sectores['Sector'],
                y=df_sectores['Camareras'],
                marker_color=['#90CAF9', '#FFE082', '#FF8A80']
            ))
            fig.update_layout(
                title='Habitaciones vs Camareras por Sector',
                barmode='group',
                yaxis_title='Cantidad'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_graf_sect2:
            # Gráfico de carga (tiempo por camarera)
            fig = px.bar(
                df_sectores,
                x='Sector',
                y='Tiempo/cam (min)',
                color='Sector',
                title='Tiempo estimado por camarera (min)',
                color_discrete_map={
                    'Bajo (Pl 2-15)': '#1E88E5',
                    'Medio (Pl 16-30)': '#FFC107',
                    'Alto (Pl 31-52)': '#DC143C'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== TABLA DE SECTORES =====
        st.subheader("📋 Detalle por Sector")
        
        # Formatear tabla para mejor visualización
        df_sectores_display = df_sectores.copy()
        df_sectores_display['Tiempo total (min)'] = df_sectores_display['Tiempo total (min)'].astype(int)
        df_sectores_display['Tiempo/cam (min)'] = df_sectores_display['Tiempo/cam (min)'].astype(int)
        
        st.dataframe(
            df_sectores_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Sector': 'Sector',
                'Habitaciones': '🏨 Habitaciones',
                'Camareras': '👤 Camareras',
                'Hab/Cam': '📊 Hab/Cam',
                'Tiempo total (min)': '⏱️ Tiempo total',
                'Tiempo/cam (min)': '⏱️ Tiempo/cam'
            }
        )
        
        st.markdown("---")
        
        # ===== GRÁFICOS ADICIONALES =====
        st.subheader("📈 Distribución por planta")
        
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            if 'planta' in df.columns:
                planta_counts = df['planta'].value_counts().sort_index()
                fig = px.bar(
                    x=planta_counts.index,
                    y=planta_counts.values,
                    labels={'x': 'Planta', 'y': 'Habitaciones'},
                    title='Habitaciones por planta',
                    color_discrete_sequence=['#1E88E5']
                )
                # Añadir líneas de separación de sectores
                fig.add_vline(x=15.5, line_dash="dash", line_color="gray", annotation_text="Bajo/Medio")
                fig.add_vline(x=30.5, line_dash="dash", line_color="gray", annotation_text="Medio/Alto")
                st.plotly_chart(fig, use_container_width=True)
        
        with col_graf2:
            if 'prob_late' in df.columns:
                fig = px.histogram(
                    df, 
                    x='prob_late', 
                    nbins=20,
                    title='Distribución de probabilidades Late Checkout',
                    labels={'prob_late': 'Probabilidad'},
                    color_discrete_sequence=['#DC143C']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA CON STAND BY
# =============================================================================

elif pagina == "🧹 Camarera":
    st.title("🧹 App Camarera - Hotel Gran Bali")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ El gerente debe cargar el PMS primero")
    elif not st.session_state.asignacion_por_camarera:
        st.warning("⚠️ No hay asignación disponible")
    else:
        if st.session_state.camarera_actual is None:
            st.subheader("👤 Selecciona tu perfil")
            camareras = list(st.session_state.asignacion_por_camarera.keys())
            st.session_state.camarera_actual = st.selectbox(
                "Nombre:",
                camareras,
                index=None,
                placeholder="Elige tu nombre...",
                key="select_camarera_principal"
            )
            
            if st.session_state.camarera_actual:
                st.session_state.habitaciones_completadas = []
                # Inicializar lista de stand by para esta camarera si no existe
                if st.session_state.camarera_actual not in st.session_state.standby_por_camarera:
                    st.session_state.standby_por_camarera[st.session_state.camarera_actual] = []
                st.rerun()
        else:
            # Obtener asignación de esta camarera
            df_asignadas = st.session_state.asignacion_por_camarera.get(
                st.session_state.camarera_actual, 
                pd.DataFrame()
            )
            
            # Obtener lista de stand by de esta camarera
            if st.session_state.camarera_actual not in st.session_state.standby_por_camarera:
                st.session_state.standby_por_camarera[st.session_state.camarera_actual] = []
            
            standby_list = st.session_state.standby_por_camarera[st.session_state.camarera_actual]
            
            # Calcular totales
            total_asignadas = len(df_asignadas)
            completadas = len(st.session_state.habitaciones_completadas)
            pendientes = total_asignadas - completadas - len(standby_list)
            
            # Mostrar información de la camarera con contador de Stand By
            col_info1, col_info2, col_info3, col_info4 = st.columns([2, 2, 1, 1])
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                if len(df_asignadas) > 0:
                    plantas_unicas = sorted(df_asignadas['planta'].unique())
                    # Verificar si son adyacentes
                    son_adyacentes = all(
                        plantas_unicas[i+1] - plantas_unicas[i] == 1 
                        for i in range(len(plantas_unicas)-1)
                    ) if len(plantas_unicas) > 1 else True
                    
                    if son_adyacentes and len(plantas_unicas) > 1:
                        st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas) ✅")
                    else:
                        st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas)")
                else:
                    st.info("📌 Sin asignación")
            with col_info3:
                st.metric("⏸️ Stand By", len(standby_list))
            with col_info4:
                if st.button("🔄 Cambiar", key="btn_cambiar_usuario_principal"):
                    st.session_state.camarera_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.rerun()
            
            st.markdown("---")
            
            # Barra de progreso
            if total_asignadas > 0:
                progreso_total = completadas / total_asignadas
                st.progress(progreso_total, text=f"**{completadas}/{total_asignadas}** habitaciones completadas")
            
            # ===== SECCIÓN 1: LIMPIEZA EN CURSO =====
            if st.session_state.cronometro_activo and st.session_state.habitacion_actual is not None:
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
                            time.sleep(1)
                            st.rerun()
                    
                    # Reportar incidencia o mantenimiento (actualizado con nuevos tipos)
                    with st.expander("⚠️ Reportar problema"):
                        tipo_reporte = st.selectbox(
                            "Tipo de problema",
                            ["Avería (Mantenimiento)", "Falta suministro", "Muy sucia", "Ocupada", "Otro"],
                            key="tipo_reporte_cron"
                        )
                        desc_reporte = st.text_area("Descripción", key="desc_reporte_cron")
                        if st.button("Enviar reporte", key="btn_reporte_cron", use_container_width=True):
                            hab_id = hab['habitacion_id']
                            
                            # Avería va a Mantenimiento
                            if tipo_reporte == "Avería (Mantenimiento)":
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
                                # No se mueve a completadas ni a stand by
                            
                            # Estos tipos van a Stand By
                            elif tipo_reporte in ["Falta suministro", "Muy sucia", "Ocupada"]:
                                # Añadir a la lista de stand by
                                if hab_id not in standby_list:
                                    standby_list.append(hab_id)
                                    st.session_state.standby_por_camarera[st.session_state.camarera_actual] = standby_list
                                
                                # Reportar como incidencia
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
                            
                            # Otro también va a Incidencias pero no a stand by
                            else:  # Otro
                                st.session_state.incidencias.append({
                                    'habitacion': int(hab_id),
                                    'planta': int(hab['planta']),
                                    'tipo': tipo_reporte,
                                    'descripcion': desc_reporte,
                                    'timestamp': datetime.now().strftime("%H:%M"),
                                    'fecha': datetime.now().strftime("%d/%m/%Y"),
                                    'reportado_por': st.session_state.camarera_actual
                                })
                                st.success("⚠️ Incidencia reportada")
                            
                            # Reiniciar cronómetro
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            time.sleep(1)
                            st.rerun()
                    
                    st.markdown("---")
            
            # ===== SECCIÓN 2: STAND BY (si hay) =====
            if standby_list:
                with st.container():
                    st.subheader(f"⏸️ Stand By ({len(standby_list)})")
                    st.caption("Habitaciones pendientes por problemas (Falta suministro, Muy sucia, Ocupada)")
                    
                    # Obtener datos de las habitaciones en stand by
                    df_standby = df_asignadas[df_asignadas['habitacion_id'].isin(standby_list)].copy()
                    
                    for i, (_, row) in enumerate(df_standby.iterrows()):
                        with st.container():
                            cols = st.columns([3, 2, 2, 3])
                            
                            with cols[0]:
                                st.markdown(f"⏸️ **Hab {int(row['habitacion_id'])}**")
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
                                # Botón para marcar como completada desde stand by
                                if st.button(
                                    f"✅ Completar", 
                                    key=f"btn_completar_standby_{row['habitacion_id']}_{i}",
                                    use_container_width=True
                                ):
                                    # Quitar de stand by
                                    standby_list.remove(row['habitacion_id'])
                                    st.session_state.standby_por_camarera[st.session_state.camarera_actual] = standby_list
                                    
                                    # Añadir a completadas
                                    st.session_state.habitaciones_completadas.append(row['habitacion_id'])
                                    
                                    st.success(f"✅ Habitación {int(row['habitacion_id'])} completada")
                                    time.sleep(1)
                                    st.rerun()
                            
                            st.divider()
                    st.markdown("---")
            
            # ===== SECCIÓN 3: HABITACIONES PENDIENTES =====
            if total_asignadas > 0:
                # Excluir completadas y stand by
                df_pendientes = df_asignadas[
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_completadas + standby_list)
                ].copy()
                
                if 'late_checkout_pred' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['late_checkout_pred', 'habitacion_id'], 
                        ascending=[False, False]
                    )
                
                st.markdown(f"### Pendientes ({pendientes} restantes)")
                
                if pendientes == 0 and not standby_list:
                    st.success("🎉 ¡Has completado todas tus habitaciones!")
                    st.balloons()
                elif pendientes > 0:
                    # Usar un contador para generar keys únicos
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
                                # Botón deshabilitado si hay limpieza en curso
                                if st.session_state.cronometro_activo:
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
# VISTA INCIDENCIAS (Falta suministro, Muy sucia, Ocupada, Otro)
# =============================================================================

elif pagina == "⚠️ Incidencias":
    st.title("⚠️ Panel de Incidencias Operativas")
    st.caption("Problemas de: Falta suministro, Muy sucia, Ocupada, Otro")
    
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
# VISTA MANTENIMIENTO (Averías técnicas)
# =============================================================================

elif pagina == "🔧 Mantenimiento":
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
    
    # Estadísticas de mantenimiento
    if st.session_state.mantenimiento:
        st.markdown("---")
        st.subheader("📊 Estadísticas de Mantenimiento")
        
        # Contar por tipo
        tipos = {}
        for inc in st.session_state.mantenimiento:
            tipo = inc['tipo']
            tipos[tipo] = tipos.get(tipo, 0) + 1
        
        col_mant1, col_mant2 = st.columns(2)
        with col_mant1:
            st.metric("Total averías", len(st.session_state.mantenimiento))
        with col_mant2:
            st.metric("Pendientes", len(st.session_state.mantenimiento))

# =============================================================================
# VISTA DATASET
# =============================================================================

elif pagina == "📋 Dataset":
    st.title("📋 Dataset Enriquecido")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ Primero carga un archivo PMS en la vista Gerente")
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
