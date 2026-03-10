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
# VISTA GERENTE - CON HEATMAP, OCUPACIÓN Y CARGA DE TRABAJO
# =============================================================================

elif selected == "📊 Gerente":
    st.title("📊 Dashboard Gerente - Hotel Gran Bali")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ Carga un archivo PMS desde el menú lateral")
    else:
        df = st.session_state.df_pms
        total_habitaciones = 446
        
        # ===== MÉTRICAS PRINCIPALES =====
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        with col_metric1:
            ocupacion = len(df) / total_habitaciones * 100
            st.markdown(
                f"""
                <div style="
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    border: 4px solid #1E88E5;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    background: transparent;
                ">
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: bold;">{ocupacion:.1f}%</div>
                        <div style="font-size: 14px;">Ocupación</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_metric2:
            st.metric("Habitaciones", len(df))
        with col_metric3:
            st.metric("Camareras", "35")
        with col_metric4:
            st.metric("Stand By", len(st.session_state.habitaciones_standby))
        
        st.markdown("---")
        
        # ===== HEATMAP DE HABITACIONES POR SECTOR =====
        st.subheader("🗺️ Estado de Habitaciones")
        
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
            return "#FFFFFF"  # Blanco (pendiente)
        
        # Crear pestañas por sector
        tab1, tab2, tab3 = st.tabs(["🔵 Sector Bajo", "🟡 Sector Medio", "🔴 Sector Alto"])
        
        for tab, (sector_nombre, plantas) in zip([tab1, tab2, tab3], sectores.items()):
            with tab:
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
        
        # ===== LEYENDA DE COLORES =====
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
        
        st.markdown("---")
        
        # ===== CARGA DE TRABAJO POR CAMARERA =====
        st.subheader("📊 Carga de trabajo por camarera")
        
        if st.session_state.asignacion_por_camarera:
            # Preparar datos para el gráfico
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
            umbral_alto = media_hab * 1.2  # 20% por encima de la media
            
            # Crear gráfico de barras
            fig = px.bar(
                df_carga,
                x='Camarera',
                y='Habitaciones',
                title='Habitaciones asignadas por camarera',
                color='Habitaciones',
                color_continuous_scale=['#4CAF50', '#FFC107', '#FF4444'],
                range_color=[0, df_carga['Habitaciones'].max()]
            )
            
            # Añadir línea de media
            fig.add_hline(
                y=media_hab,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Media: {media_hab:.1f}",
                annotation_position="top right"
            )
            
            # Marcar en rojo las que superan el umbral
            colores_barras = []
            for hab in df_carga['Habitaciones']:
                if hab > umbral_alto:
                    colores_barras.append('#FF4444')
                else:
                    colores_barras.append('#1E88E5')
            
            fig.update_traces(marker_color=colores_barras)
            
            fig.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="Número de habitaciones",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar camareras con sobrecarga
            sobrecargadas = df_carga[df_carga['Habitaciones'] > umbral_alto]
            if len(sobrecargadas) > 0:
                st.warning("⚠️ Camareras con sobrecarga de trabajo:")
                for _, row in sobrecargadas.iterrows():
                    st.markdown(f"- {row['Camarera']}: {row['Habitaciones']} habitaciones")
        
        st.markdown("---")
        
        # ===== DETALLE DE HABITACIONES =====
        st.subheader("📋 Detalle de habitaciones")
        
        # Añadir columna de estado para mejor visualización
        df_display = df.copy()
        estados = []
        for hab_id in df_display['habitacion_id']:
            if hab_id in st.session_state.habitaciones_completadas:
                # Verificar tipo de problema
                encontrado = False
                for inc in st.session_state.incidencias:
                    if inc['habitacion'] == hab_id:
                        estados.append(f"⚠️ {inc['tipo']}")
                        encontrado = True
                        break
                if not encontrado:
                    for mant in st.session_state.mantenimiento:
                        if mant['habitacion'] == hab_id:
                            estados.append("🔧 Mantenimiento")
                            encontrado = True
                            break
                if not encontrado:
                    estados.append("✅ Completada")
            elif hab_id in st.session_state.habitaciones_standby:
                estados.append("⏸️ Stand By")
            else:
                estados.append("⏳ Pendiente")
        
        df_display['Estado'] = estados
        
        st.dataframe(
            df_display[['habitacion_id', 'planta', 'clase_checkout', 'tiempo_estimado', 'Estado']].head(100),
            use_container_width=True,
            height=400,
            column_config={
                'habitacion_id': 'Habitación',
                'planta': 'Planta',
                'clase_checkout': 'Tipo',
                'tiempo_estimado': 'Tiempo est. (min)',
                'Estado': 'Estado'
            }
        )

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
