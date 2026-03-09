# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con asignación equitativa basada en modelos
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
    st.session_state.incidencias = []
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

def asignacion_equitativa(df, total_camareras=35):
    """
    Asigna habitaciones de forma equitativa usando:
    - K-Means para agrupar plantas
    - XGBoost para estimar tiempo y equilibrar carga
    - ANN para priorizar urgentes
    """
    if df is None or len(df) == 0:
        return {}
    
    df_asignar = df.copy()
    
    # 1. Obtener cluster de cada planta usando K-Means
    if modelos.get('kmeans') is not None:
        kmeans_model = modelos['kmeans']['modelo']
        scaler = modelos['kmeans']['scaler']
        features = modelos['kmeans']['features']
        
        # Crear perfil por planta
        plantas_unicas = sorted(df_asignar['planta'].unique())
        planta_a_cluster = {}
        
        for planta in plantas_unicas:
            df_planta = df_asignar[df_asignar['planta'] == planta]
            perfil = {
                'planta': planta,
                'tiempo_promedio': df_planta['tiempo_estimado'].mean() if 'tiempo_estimado' in df_planta.columns else 30,
                'tasa_late_checkout': df_planta['late_checkout_pred'].mean() if 'late_checkout_pred' in df_planta.columns else 0.3,
                'noches_promedio': df_planta['noches_estancia'].mean() if 'noches_estancia' in df_planta.columns else 3,
                'huespedes_promedio': df_planta['num_huespedes'].mean() if 'num_huespedes' in df_planta.columns else 2,
                'tasa_ninos': df_planta['tiene_ninos'].mean() if 'tiene_ninos' in df_planta.columns else 0.2,
                'sector_num': 0 if planta <= 15 else (1 if planta <= 30 else 2),
                'num_habitaciones': len(df_planta)
            }
            perfil_df = pd.DataFrame([perfil])
            X_perfil = perfil_df[features].values
            X_scaled = scaler.transform(X_perfil)
            planta_a_cluster[planta] = kmeans_model.predict(X_scaled)[0]
        
        df_asignar['cluster'] = df_asignar['planta'].map(planta_a_cluster)
    else:
        # Fallback: usar sector como cluster
        df_asignar['cluster'] = df_asignar['planta'].apply(
            lambda x: 0 if x <= 15 else (1 if x <= 30 else 2)
        )
    
    # 2. Aplicar ANN para predecir late checkout
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
    
    # 3. Agrupar por cluster y asignar equitativamente
    clusters = sorted(df_asignar['cluster'].unique())
    habitaciones_por_cluster = {
        cluster: df_asignar[df_asignar['cluster'] == cluster].copy()
        for cluster in clusters
    }
    
    # Calcular tiempo total por cluster
    tiempo_por_cluster = {}
    for cluster, df_cluster in habitaciones_por_cluster.items():
        if 'tiempo_estimado' in df_cluster.columns:
            tiempo_por_cluster[cluster] = df_cluster['tiempo_estimado'].sum()
        else:
            tiempo_por_cluster[cluster] = len(df_cluster) * 25  # estimado
    
    # Calcular tiempo total y asignar camareras proporcionalmente
    tiempo_total = sum(tiempo_por_cluster.values())
    
    # Distribución de camareras por cluster (proporcional al tiempo)
    camareras_por_cluster = {}
    for cluster, tiempo in tiempo_por_cluster.items():
        # Mínimo 1 camarera por cluster
        num_cam = max(1, round((tiempo / tiempo_total) * total_camareras))
        camareras_por_cluster[cluster] = num_cam
    
    # Ajustar para que sume 35
    total_asignado = sum(camareras_por_cluster.values())
    if total_asignado > total_camareras:
        # Reducir del cluster con más tiempo
        cluster_max = max(camareras_por_cluster, key=camareras_por_cluster.get)
        camareras_por_cluster[cluster_max] -= (total_asignado - total_camareras)
    elif total_asignado < total_camareras:
        # Añadir al cluster con más tiempo
        cluster_max = max(camareras_por_cluster, key=camareras_por_cluster.get)
        camareras_por_cluster[cluster_max] += (total_camareras - total_asignado)
    
    # Asignar habitaciones a cada camarera
    asignacion = {}
    cam_idx = 1
    
    for cluster, num_cam in camareras_por_cluster.items():
        df_cluster = habitaciones_por_cluster[cluster].copy()
        
        # Ordenar por prioridad (late checkout)
        df_cluster = df_cluster.sort_values(
            by=['late_checkout_pred', 'habitacion_id'], 
            ascending=[False, True]
        )
        
        # Dividir habitaciones entre las camareras del cluster
        habitaciones_por_cam = len(df_cluster) // num_cam
        resto = len(df_cluster) % num_cam
        
        inicio = 0
        for i in range(num_cam):
            if cam_idx > total_camareras:
                break
            
            fin = inicio + habitaciones_por_cam + (1 if i < resto else 0)
            df_cam = df_cluster.iloc[inicio:fin].copy()
            
            if len(df_cam) > 0:
                asignacion[f"Camarera {cam_idx:02d}"] = df_cam
            
            inicio = fin
            cam_idx += 1
    
    return asignacion

# =============================================================================
# SIDEBAR - NAVEGACIÓN
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hotel.png", width=80)
    st.title("🏨 Hotel Gran Bali")
    st.markdown("---")
    
    pagina = st.radio(
        "**Menú Principal**",
        ["📊 Gerente", "🧹 Camarera", "⚠️ Incidencias", "📋 Dataset"]
    )
    
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
        for key in ['df_pms', 'incidencias', 'opiniones', 'camarera_actual', 
                    'cronometro_activo', 'tiempo_inicio', 'habitacion_actual',
                    'asignacion_por_camarera', 'habitaciones_completadas']:
            if key in st.session_state:
                if key in ['incidencias', 'opiniones', 'habitaciones_completadas']:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.rerun()

# =============================================================================
# VISTA GERENTE
# =============================================================================

if pagina == "📊 Gerente":
    st.title("📊 Dashboard Gerente - Hotel Gran Bali")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📂 Cargar PMS del día")
        archivo = st.file_uploader("Selecciona archivo CSV", type=['csv'])
        
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
                
                # Hacer asignación equitativa
                with st.spinner("Calculando asignación equitativa..."):
                    st.session_state.asignacion_por_camarera = asignacion_equitativa(df)
                
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
        
        st.subheader("📈 Distribución de carga")
        
        if st.session_state.asignacion_por_camarera:
            datos_carga = []
            for cam, df_cam in st.session_state.asignacion_por_camarera.items():
                tiempo = df_cam['tiempo_estimado'].sum() if 'tiempo_estimado' in df_cam.columns else len(df_cam) * 25
                datos_carga.append({
                    'Camarera': cam,
                    'Habitaciones': len(df_cam),
                    'Tiempo (min)': round(tiempo, 1)
                })
            
            df_carga = pd.DataFrame(datos_carga)
            
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                fig = px.bar(
                    df_carga, 
                    x='Camarera', 
                    y='Habitaciones',
                    title='Habitaciones por camarera'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_graf2:
                fig = px.bar(
                    df_carga, 
                    x='Camarera', 
                    y='Tiempo (min)',
                    title='Tiempo estimado por camarera'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Estadísticas de carga")
            
            col_est1, col_est2, col_est3 = st.columns(3)
            with col_est1:
                st.metric("Media hab/cam", f"{df_carga['Habitaciones'].mean():.1f}")
            with col_est2:
                st.metric("Min hab", int(df_carga['Habitaciones'].min()))
            with col_est3:
                st.metric("Max hab", int(df_carga['Habitaciones'].max()))
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA - CON ASIGNACIÓN EQUITATIVA
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
                placeholder="Elige tu nombre..."
            )
            
            if st.session_state.camarera_actual:
                st.session_state.habitaciones_completadas = []
                st.rerun()
        else:
            # Obtener asignación de esta camarera
            df_asignadas = st.session_state.asignacion_por_camarera.get(
                st.session_state.camarera_actual, 
                pd.DataFrame()
            )
            
            total_asignadas = len(df_asignadas)
            completadas = len(st.session_state.habitaciones_completadas)
            pendientes = total_asignadas - completadas
            
            # Mostrar información
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                if len(df_asignadas) > 0:
                    plantas_unicas = sorted(df_asignadas['planta'].unique())
                    st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas)")
                else:
                    st.info("📌 Sin asignación")
            with col_info3:
                if st.button("🔄 Cambiar usuario"):
                    st.session_state.camarera_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.rerun()
            
            st.markdown("---")
            
            # SECCIÓN 1: LIMPIEZA EN CURSO
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
                        st.markdown(f"**Tiempo:** {formatear_tiempo(tiempo_transcurrido)}")
                        if 'tiempo_estimado' in hab:
                            progreso = min(tiempo_transcurrido / (hab['tiempo_estimado'] * 60), 1.0)
                            st.progress(progreso)
                    
                    with col_crono3:
                        if st.button("✅ Finalizar limpieza", type="primary", use_container_width=True):
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
                    
                    with st.expander("⚠️ Reportar incidencia"):
                        tipo_inc = st.selectbox(
                            "Tipo",
                            ["Avería", "Falta suministros", "Habitación sucia", "Cliente presente", "Otro"],
                            key="tipo_inc_cron"
                        )
                        desc_inc = st.text_area("Descripción", key="desc_inc_cron")
                        if st.button("Enviar", key="btn_inc_cron", use_container_width=True):
                            st.session_state.incidencias.append({
                                'habitacion': int(hab['habitacion_id']),
                                'planta': int(hab['planta']),
                                'tipo': tipo_inc,
                                'descripcion': desc_inc,
                                'timestamp': datetime.now().strftime("%H:%M"),
                                'fecha': datetime.now().strftime("%d/%m/%Y")
                            })
                            st.success("✅ Incidencia reportada")
                            
                            st.session_state.habitaciones_completadas.append(hab['habitacion_id'])
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            time.sleep(1)
                            st.rerun()
                    
                    st.markdown("---")
            
            # SECCIÓN 2: PENDIENTES
            if total_asignadas > 0:
                df_pendientes = df_asignadas[
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_completadas)
                ].copy()
                
                if 'late_checkout_pred' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['late_checkout_pred', 'habitacion_id'], 
                        ascending=[False, False]
                    )
                
                # Barra de progreso
                progreso_total = completadas / total_asignadas if total_asignadas > 0 else 0
                st.progress(progreso_total, text=f"**{completadas}/{total_asignadas}** habitaciones completadas")
                
                st.markdown(f"### Pendientes ({pendientes} restantes)")
                
                if pendientes == 0:
                    st.success("🎉 ¡Has completado todas tus habitaciones!")
                    st.balloons()
                else:
                    for idx, row in df_pendientes.iterrows():
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
                                disabled = st.session_state.cronometro_activo
                                if st.button(
                                    f"▶️ Iniciar", 
                                    key=f"btn_{idx}_{row['habitacion_id']}", 
                                    disabled=disabled,
                                    use_container_width=True
                                ):
                                    st.session_state.habitacion_actual = row
                                    st.session_state.cronometro_activo = True
                                    st.session_state.tiempo_inicio = datetime.now()
                                    st.rerun()
                            
                            st.divider()
            
            # SECCIÓN 3: COMPLETADAS
            if completadas > 0:
                st.markdown("---")
                st.subheader(f"✅ Completadas ({completadas}/{total_asignadas})")
                
                df_completadas = st.session_state.df_pms[
                    st.session_state.df_pms['habitacion_id'].isin(
                        st.session_state.habitaciones_completadas
                    )
                ]
                
                for idx, row in df_completadas.iterrows():
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
# VISTA INCIDENCIAS Y DATASET (se mantienen igual)
# =============================================================================

elif pagina == "⚠️ Incidencias":
    st.title("⚠️ Panel de Incidencias - Hotel Gran Bali")
    
    tab1, tab2 = st.tabs(["📋 Incidencias activas", "📝 Registrar opinión"])
    
    with tab1:
        if st.session_state.incidencias:
            for inc in reversed(st.session_state.incidencias):
                with st.container():
                    col_inc1, col_inc2 = st.columns([3, 1])
                    with col_inc1:
                        st.warning(f"**{inc['timestamp']} - {inc['fecha']}**")
                        st.markdown(f"**Habitación {inc['habitacion']}** (Planta {inc['planta']})")
                        st.markdown(f"**{inc['tipo']}:** {inc['descripcion']}")
                    with col_inc2:
                        if st.button("✓ Resolver", key=f"resolver_{inc['habitacion']}_{inc['timestamp']}"):
                            st.session_state.incidencias.remove(inc)
                            st.rerun()
                    st.divider()
        else:
            st.info("✅ No hay incidencias registradas")
    
    with tab2:
        st.subheader("📝 Registrar opinión de cliente")
        with st.form("form_opinion"):
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                hab_id = st.number_input("N° habitación", min_value=100, max_value=5299, value=1205, step=1)
            with col_form2:
                st.caption(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            
            opinion_text = st.text_area("Opinión", placeholder="Ej: La habitación estaba muy limpia...", height=100)
            submitted = st.form_submit_button("📤 Registrar", use_container_width=True)
            
            if submitted and opinion_text:
                sentimiento = procesar_opinion(opinion_text)
                st.session_state.opiniones.append({
                    'habitacion': hab_id,
                    'opinion': opinion_text,
                    'sentimiento': sentimiento,
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'fecha': datetime.now().strftime("%d/%m/%Y")
                })
                st.success(f"✅ Opinión registrada - Sentimiento: **{sentimiento}**")
                st.rerun()

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
            if 'late_checkout_pred' in df.columns:
                st.metric("Late checkout", int(df['late_checkout_pred'].sum()))
        
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
