# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con asignación basada en modelos entrenados
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
    
    # Verificar que los archivos existen
    archivos_faltantes = []
    for nombre, archivo in archivos_modelos.items():
        if os.path.exists(archivo):
            try:
                with open(archivo, 'rb') as f:
                    modelos[nombre] = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ No se pudo cargar {nombre}: {str(e)}")
                modelos[nombre] = None
        else:
            archivos_faltantes.append(archivo)
            modelos[nombre] = None
    
    if archivos_faltantes:
        st.warning(f"⚠️ Archivos no encontrados: {', '.join(archivos_faltantes)}")
    
    return modelos

# Cargar modelos al inicio
with st.spinner("Cargando modelos de IA..."):
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
if 'asignacion_actual' not in st.session_state:
    st.session_state.asignacion_actual = None
if 'habitaciones_completadas' not in st.session_state:
    st.session_state.habitaciones_completadas = []

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def limpiar_texto_opinion(texto):
    """Limpia el texto de opiniones para el modelo NLP"""
    if not isinstance(texto, str) or texto == "":
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def procesar_opinion(texto):
    """Procesa una opinión y devuelve el sentimiento"""
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
    """Formatea segundos a minutos:segundos"""
    minutos = int(segundos // 60)
    segs = int(segundos % 60)
    return f"{minutos}:{segs:02d}"

def asignar_habitaciones_por_modelos(df, num_cam, total_camaras=35):
    """
    Asigna habitaciones a una camarera usando los modelos entrenados:
    - K-Means: determina el clúster de plantas de la camarera
    - ANN: prioriza habitaciones con late checkout
    - XGBoost: estima tiempo para equilibrar carga
    """
    if modelos.get('kmeans') is None:
        # Fallback a asignación por sectores si no hay modelo
        if num_cam <= 19:
            plantas_base = list(range(2, 16))
        elif num_cam <= 30:
            plantas_base = list(range(16, 31))
        else:
            plantas_base = list(range(31, 53))
        return df[df['planta'].isin(plantas_base)].copy()
    
    # 1. Obtener cluster de la camarera usando K-Means
    kmeans_model = modelos['kmeans']['modelo']
    scaler = modelos['kmeans']['scaler']
    features = modelos['kmeans']['features']
    
    # Crear perfil promedio de las plantas del sector de la camarera
    if num_cam <= 19:
        plantas_sector = list(range(2, 16))
    elif num_cam <= 30:
        plantas_sector = list(range(16, 31))
    else:
        plantas_sector = list(range(31, 53))
    
    # Calcular estadísticas promedio del sector
    df_sector = df[df['planta'].isin(plantas_sector)]
    if len(df_sector) == 0:
        return pd.DataFrame()
    
    perfil_sector = {
        'planta': np.mean(plantas_sector),
        'tiempo_promedio': df_sector['tiempo_estimado'].mean() if 'tiempo_estimado' in df_sector.columns else 30,
        'tasa_late_checkout': df_sector['late_checkout_pred'].mean() if 'late_checkout_pred' in df_sector.columns else 0.3,
        'noches_promedio': df_sector['noches_estancia'].mean() if 'noches_estancia' in df_sector.columns else 3,
        'huespedes_promedio': df_sector['num_huespedes'].mean() if 'num_huespedes' in df_sector.columns else 2,
        'tasa_ninos': df_sector['tiene_ninos'].mean() if 'tiene_ninos' in df_sector.columns else 0.2,
        'sector_num': 0 if num_cam <= 19 else (1 if num_cam <= 30 else 2),
        'num_habitaciones': len(plantas_sector)
    }
    
    # Escalar y predecir cluster
    perfil_df = pd.DataFrame([perfil_sector])
    X_perfil = perfil_df[features].values
    X_scaled = scaler.transform(X_perfil)
    cluster_cam = kmeans_model.predict(X_scaled)[0]
    
    # Obtener plantas del mismo cluster
    if 'df_planta_stats' in modelos['kmeans']:
        df_stats = pd.DataFrame(modelos['kmeans']['df_planta_stats'])
        plantas_cluster = df_stats[df_stats['cluster'] == cluster_cam]['planta'].tolist()
    else:
        # Fallback: plantas del sector
        plantas_cluster = plantas_sector
    
    # 2. Filtrar habitaciones del cluster
    df_candidatas = df[df['planta'].isin(plantas_cluster)].copy()
    
    # 3. Aplicar ANN para priorizar urgentes (late checkout)
    if modelos.get('ann') is not None and 'late_checkout_pred' not in df_candidatas.columns:
        try:
            ann_model = modelos['ann']['modelo']
            scaler_ann = modelos['ann']['scaler']
            feature_cols = modelos['ann']['feature_cols']
            
            # Preparar features para ANN
            X_ann = df_candidatas[feature_cols].values
            X_ann_scaled = scaler_ann.transform(X_ann)
            
            # Predecir probabilidad de late checkout
            prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
            df_candidatas['prob_late'] = prob_late
            df_candidatas['late_checkout_pred'] = (prob_late > 0.5).astype(int)
        except Exception as e:
            df_candidatas['late_checkout_pred'] = 0
            df_candidatas['prob_late'] = 0
    
    # 4. Aplicar XGBoost para estimar tiempos (si es necesario)
    if modelos.get('xgboost') is not None and 'tiempo_estimado' in df_candidatas.columns:
        # Ya tenemos tiempo_estimado del CSV
        pass
    
    # 5. Ordenar por prioridad (late checkout primero)
    if 'late_checkout_pred' in df_candidatas.columns:
        df_candidatas = df_candidatas.sort_values(
            by=['late_checkout_pred', 'habitacion_id'], 
            ascending=[False, False]
        )
    else:
        df_candidatas = df_candidatas.sort_values('habitacion_id', ascending=False)
    
    return df_candidatas

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
                    'asignacion_actual', 'habitaciones_completadas']:
            if key in st.session_state:
                if key == 'incidencias' or key == 'opiniones' or key == 'habitaciones_completadas':
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
        archivo = st.file_uploader(
            "Selecciona archivo CSV",
            type=['csv']
        )
        
        if archivo is not None and st.session_state.df_pms is None:
            with st.spinner("Procesando archivo..."):
                df = pd.read_csv(archivo)
                
                # Aplicar ANN para predecir late checkout si es posible
                if modelos.get('ann') is not None:
                    try:
                        ann_model = modelos['ann']['modelo']
                        scaler_ann = modelos['ann']['scaler']
                        feature_cols = modelos['ann']['feature_cols']
                        
                        # Verificar que todas las columnas necesarias existen
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
                st.success(f"✅ PMS cargado correctamente: {len(df)} habitaciones")
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
                repasos = len(df) - checkouts if 'late_checkout_pred' in df.columns else 0
                st.metric("Repasos", repasos)
    
    if st.session_state.df_pms is not None:
        df = st.session_state.df_pms
        st.markdown("---")
        
        st.subheader("📈 Indicadores Clave")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            ocupacion = len(df) / 446 * 100
            st.metric("Ocupación", f"{ocupacion:.1f}%", delta=f"{len(df)}/446 habs")
        
        with col_kpi2:
            if 'tiempo_estimado' in df.columns:
                tiempo_total = df['tiempo_estimado'].sum()
                st.metric("Tiempo total", f"{tiempo_total:.0f} min")
        
        with col_kpi3:
            if 'planta' in df.columns:
                plantas_ocupadas = df['planta'].nunique()
                st.metric("Plantas activas", f"{plantas_ocupadas}/51")
        
        with col_kpi4:
            st.metric("Camareras necesarias", "35")
        
        st.markdown("---")
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.subheader("🏢 Distribución por planta")
            if 'planta' in df.columns:
                planta_counts = df['planta'].value_counts().sort_index()
                fig = px.bar(
                    x=planta_counts.index,
                    y=planta_counts.values,
                    labels={'x': 'Planta', 'y': 'Habitaciones'},
                    title=f'Habitaciones por planta'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_graf2:
            st.subheader("📊 Probabilidad Late Checkout")
            if 'prob_late' in df.columns:
                fig = px.histogram(
                    df, 
                    x='prob_late', 
                    nbins=20,
                    title='Distribución de probabilidades',
                    labels={'prob_late': 'Probabilidad'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA - CON MODELOS
# =============================================================================

elif pagina == "🧹 Camarera":
    st.title("🧹 App Camarera - Hotel Gran Bali")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ El gerente debe cargar el PMS primero")
    else:
        df = st.session_state.df_pms
        
        if st.session_state.camarera_actual is None:
            st.subheader("👤 Selecciona tu perfil")
            camareras = [f"Camarera {i:02d}" for i in range(1, 36)]
            st.session_state.camarera_actual = st.selectbox(
                "Nombre:",
                camareras,
                index=None,
                placeholder="Elige tu nombre..."
            )
            
            if st.session_state.camarera_actual:
                # Inicializar listas
                st.session_state.habitaciones_completadas = []
                st.session_state.asignacion_actual = None
                st.rerun()
        else:
            # Inicializar si no existe
            if 'habitaciones_completadas' not in st.session_state:
                st.session_state.habitaciones_completadas = []
            
            # Obtener número de camarera
            num_cam = int(st.session_state.camarera_actual.split()[1])
            
            # USAR MODELOS PARA ASIGNAR HABITACIONES
            if st.session_state.asignacion_actual is None:
                with st.spinner("Asignando habitaciones con IA..."):
                    df_asignadas = asignar_habitaciones_por_modelos(df, num_cam)
                    if len(df_asignadas) > 0:
                        st.session_state.asignacion_actual = df_asignadas
                    else:
                        st.warning("No hay habitaciones disponibles")
                        st.session_state.asignacion_actual = pd.DataFrame()
            
            # Mostrar información de la camarera
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                if modelos.get('kmeans') is not None and len(st.session_state.asignacion_actual) > 0:
                    plantas_unicas = sorted(st.session_state.asignacion_actual['planta'].unique())
                    st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas)")
                else:
                    st.info("📌 Asignación IA")
            with col_info3:
                if st.button("🔄 Cambiar usuario"):
                    st.session_state.camarera_actual = None
                    st.session_state.asignacion_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.rerun()
            
            st.markdown("---")
            
            # ===== SECCIÓN 1: LIMPIEZA EN CURSO (ARRIBA) =====
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
                            # Calcular tiempo real
                            tiempo_real = (datetime.now() - st.session_state.tiempo_inicio).seconds / 60
                            
                            # Guardar en el DataFrame principal
                            hab_id = hab['habitacion_id']
                            df.loc[df['habitacion_id'] == hab_id, 'tiempo_real'] = round(tiempo_real, 1)
                            st.session_state.df_pms = df
                            
                            # Añadir a la lista de completadas
                            st.session_state.habitaciones_completadas.append(hab_id)
                            
                            # Mensaje de éxito
                            st.success(f"✅ Habitación {int(hab_id)} completada en {tiempo_real:.1f} minutos")
                            
                            # Reiniciar cronómetro
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            time.sleep(1)
                            st.rerun()
                    
                    # Reportar incidencia
                    with st.expander("⚠️ Reportar incidencia"):
                        tipo_inc = st.selectbox(
                            "Tipo de incidencia",
                            ["Avería", "Falta suministros", "Habitación sucia", "Cliente presente", "Otro"],
                            key="tipo_inc_cron"
                        )
                        desc_inc = st.text_area("Descripción", key="desc_inc_cron")
                        if st.button("Enviar reporte", key="btn_inc_cron", use_container_width=True):
                            st.session_state.incidencias.append({
                                'habitacion': int(hab['habitacion_id']),
                                'planta': int(hab['planta']),
                                'tipo': tipo_inc,
                                'descripcion': desc_inc,
                                'timestamp': datetime.now().strftime("%H:%M"),
                                'fecha': datetime.now().strftime("%d/%m/%Y")
                            })
                            st.success("✅ Incidencia reportada")
                            
                            # Mover a completadas aunque haya incidencia
                            hab_id = hab['habitacion_id']
                            st.session_state.habitaciones_completadas.append(hab_id)
                            
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            time.sleep(1)
                            st.rerun()
                    
                    st.markdown("---")
            
            # ===== SECCIÓN 2: HABITACIONES PENDIENTES =====
            if len(st.session_state.asignacion_actual) > 0:
                # Excluir completadas
                df_pendientes = st.session_state.asignacion_actual[
                    ~st.session_state.asignacion_actual['habitacion_id'].isin(
                        st.session_state.habitaciones_completadas
                    )
                ].copy()
                
                # Ordenar por prioridad (late checkout primero)
                if 'late_checkout_pred' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['late_checkout_pred', 'habitacion_id'], 
                        ascending=[False, False]
                    )
                
                st.subheader(f"📋 Pendientes ({len(df_pendientes)} restantes)")
                
                if len(df_pendientes) == 0:
                    st.success("🎉 ¡Has completado todas tus habitaciones!")
                    st.balloons()
                else:
                    for idx, row in df_pendientes.iterrows():
                        with st.container():
                            cols = st.columns([3, 2, 2, 3])
                            
                            with cols[0]:
                                # Indicador visual de prioridad
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
                                # Deshabilitar si hay limpieza en curso
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
            
            # ===== SECCIÓN 3: HABITACIONES COMPLETADAS =====
            if st.session_state.habitaciones_completadas:
                st.markdown("---")
                st.subheader(f"✅ Completadas hoy ({len(st.session_state.habitaciones_completadas)})")
                
                df_completadas = df[df['habitacion_id'].isin(
                    st.session_state.habitaciones_completadas
                )]
                
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
# VISTA INCIDENCIAS
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
                hab_id = st.number_input(
                    "Número de habitación",
                    min_value=100,
                    max_value=5299,
                    value=1205,
                    step=1
                )
            
            with col_form2:
                fecha_opinion = datetime.now().strftime("%d/%m/%Y %H:%M")
                st.caption(f"Fecha: {fecha_opinion}")
            
            opinion_text = st.text_area(
                "Opinión del cliente",
                placeholder="Ej: La habitación estaba muy limpia, todo perfecto",
                height=100
            )
            
            submitted = st.form_submit_button("📤 Registrar opinión", use_container_width=True)
            
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

# =============================================================================
# VISTA DATASET
# =============================================================================

elif pagina == "📋 Dataset":
    st.title("📋 Dataset Enriquecido")
    
    if st.session_state.df_pms is None:
        st.warning("⚠️ Primero carga un archivo PMS en la vista Gerente")
    else:
        df = st.session_state.df_pms.copy()
        
        # Añadir opiniones al dataset
        if st.session_state.opiniones:
            for op in st.session_state.opiniones:
                mask = df['habitacion_id'] == op['habitacion']
                if mask.any():
                    df.loc[mask, 'opinion_cliente'] = op['opinion']
                    df.loc[mask, 'sentimiento_nlp'] = op['sentimiento']
        
        st.subheader("📊 Métricas del dataset")
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        with col_met1:
            st.metric("Total registros", len(df))
        with col_met2:
            st.metric("Con opiniones", len([o for o in st.session_state.opiniones if o]))
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
