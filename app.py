# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con reparto equitativo basado en modelos
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
if 'asignacion_global' not in st.session_state:
    st.session_state.asignacion_global = None
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

def asignacion_equitativa_por_modelos(df, total_camaras=35, max_por_camara=8):
    """
    Distribuye equitativamente las habitaciones entre las 35 camareras
    usando los modelos entrenados
    """
    if df is None or len(df) == 0:
        return {}
    
    df_copy = df.copy()
    
    # 1. Aplicar ANN para predecir late checkout si es posible
    if modelos.get('ann') is not None:
        try:
            ann_model = modelos['ann']['modelo']
            scaler_ann = modelos['ann']['scaler']
            feature_cols = modelos['ann']['feature_cols']
            
            # Verificar que todas las columnas necesarias existen
            cols_disponibles = [c for c in feature_cols if c in df_copy.columns]
            if len(cols_disponibles) == len(feature_cols):
                X_ann = df_copy[feature_cols].values
                X_ann_scaled = scaler_ann.transform(X_ann)
                
                prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                df_copy['prob_late'] = prob_late
                df_copy['late_checkout_pred'] = (prob_late > 0.5).astype(int)
            else:
                df_copy['late_checkout_pred'] = 0
                df_copy['prob_late'] = 0
        except Exception as e:
            df_copy['late_checkout_pred'] = 0
            df_copy['prob_late'] = 0
    else:
        df_copy['late_checkout_pred'] = 0
        df_copy['prob_late'] = 0
    
    # 2. Asignar prioridad (late checkout = 1, normal = 0)
    df_copy['prioridad'] = df_copy['late_checkout_pred']
    
    # 3. Ordenar por prioridad (urgentes primero)
    df_sorted = df_copy.sort_values(by=['prioridad', 'habitacion_id'], ascending=[False, True])
    
    # 4. Distribuir equitativamente entre camareras
    asignacion = {f"Camarera {i:02d}": [] for i in range(1, total_camaras + 1)}
    
    for idx, row in df_sorted.iterrows():
        # Buscar camarera con menos habitaciones asignadas
        cargas = {cam: len(asignacion[cam]) for cam in asignacion}
        
        # Priorizar camareras del mismo sector si es posible
        planta = row['planta']
        if planta <= 15:
            candidatas = [f"Camarera {i:02d}" for i in range(1, 20)]
        elif planta <= 30:
            candidatas = [f"Camarera {i:02d}" for i in range(20, 31)]
        else:
            candidatas = [f"Camarera {i:02d}" for i in range(31, 36)]
        
        # De las candidatas, elegir la de menor carga
        candidatas = [c for c in candidatas if c in asignacion]
        if candidatas:
            cam_elegida = min(candidatas, key=lambda c: cargas[c])
        else:
            cam_elegida = min(asignacion.keys(), key=lambda c: cargas[c])
        
        # Asignar si no supera el máximo
        if len(asignacion[cam_elegida]) < max_por_camara:
            asignacion[cam_elegida].append(row['habitacion_id'])
    
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
                    'asignacion_global', 'habitaciones_completadas']:
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
        archivo = st.file_uploader(
            "Selecciona archivo CSV",
            type=['csv']
        )
        
        if archivo is not None and st.session_state.df_pms is None:
            with st.spinner("Procesando archivo y distribuyendo habitaciones..."):
                df = pd.read_csv(archivo)
                st.session_state.df_pms = df
                
                # Calcular asignación equitativa
                with st.spinner("Distribuyendo habitaciones entre camareras..."):
                    asignacion = asignacion_equitativa_por_modelos(df)
                    st.session_state.asignacion_global = asignacion
                
                st.success(f"✅ PMS cargado correctamente: {len(df)} habitaciones distribuidas entre 35 camareras")
                st.rerun()
    
    with col2:
        if st.session_state.df_pms is not None and st.session_state.asignacion_global is not None:
            df = st.session_state.df_pms
            st.subheader("📊 Resumen del día")
            
            # Calcular métricas de distribución
            total_hab = len(df)
            total_asignadas = sum(len(v) for v in st.session_state.asignacion_global.values())
            carga_por_cam = [len(v) for v in st.session_state.asignacion_global.values()]
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Habitaciones", total_hab)
            with col_metric2:
                st.metric("Media x camarera", f"{np.mean(carga_por_cam):.1f}")
            with col_metric3:
                st.metric("Rango", f"{min(carga_por_cam)}-{max(carga_por_cam)}")
    
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
        
        # Mostrar distribución por camarera
        if st.session_state.asignacion_global is not None:
            st.subheader("👥 Distribución por camarera")
            
            # Crear DataFrame para visualización
            distribucion = []
            for cam, habs in st.session_state.asignacion_global.items():
                num_cam = int(cam.split()[1])
                if num_cam <= 19:
                    sector = "Bajo"
                elif num_cam <= 30:
                    sector = "Medio"
                else:
                    sector = "Alto"
                
                distribucion.append({
                    'Camarera': cam,
                    'Sector': sector,
                    'Habitaciones': len(habs)
                })
            
            df_dist = pd.DataFrame(distribucion)
            
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                fig = px.bar(
                    df_dist,
                    x='Camarera',
                    y='Habitaciones',
                    color='Sector',
                    title='Habitaciones por camarera',
                    color_discrete_map={'Bajo': '#4B8BFF', 'Medio': '#FFC107', 'Alto': '#FF4B4B'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_graf2:
                fig = px.pie(
                    df_dist,
                    values='Habitaciones',
                    names='Sector',
                    title='Distribución por sector'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA - CON REPARTO EQUITATIVO
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
                st.rerun()
        else:
            # Inicializar si no existe
            if 'habitaciones_completadas' not in st.session_state:
                st.session_state.habitaciones_completadas = []
            
            # Obtener habitaciones asignadas a esta camarera
            if st.session_state.asignacion_global is not None:
                habs_asignadas = st.session_state.asignacion_global.get(
                    st.session_state.camarera_actual, []
                )
                df_asignadas = df[df['habitacion_id'].isin(habs_asignadas)].copy()
            else:
                df_asignadas = pd.DataFrame()
                st.warning("No hay asignación disponible. El gerente debe cargar el PMS.")
            
            # Calcular totales
            total_asignadas = len(df_asignadas)
            completadas = len(st.session_state.habitaciones_completadas)
            pendientes = total_asignadas - completadas
            
            # Mostrar información de la camarera
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                num_cam = int(st.session_state.camarera_actual.split()[1])
                if num_cam <= 19:
                    sector = "Bajo"
                elif num_cam <= 30:
                    sector = "Medio"
                else:
                    sector = "Alto"
                st.info(f"📌 Sector: {sector}")
            with col_info3:
                if st.button("🔄 Cambiar usuario"):
                    st.session_state.camarera_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.rerun()
            
            st.markdown("---")
            
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
            if total_asignadas > 0:
                # Excluir completadas
                df_pendientes = df_asignadas[
                    ~df_asignadas['habitacion_id'].isin(
                        st.session_state.habitaciones_completadas
                    )
                ].copy()
                
                # Aplicar ANN para priorizar si no se hizo antes
                if 'late_checkout_pred' not in df_pendientes.columns and modelos.get('ann') is not None:
                    try:
                        ann_model = modelos['ann']['modelo']
                        scaler_ann = modelos['ann']['scaler']
                        feature_cols = modelos['ann']['feature_cols']
                        
                        cols_disponibles = [c for c in feature_cols if c in df_pendientes.columns]
                        if len(cols_disponibles) == len(feature_cols):
                            X_ann = df_pendientes[feature_cols].values
                            X_ann_scaled = scaler_ann.transform(X_ann)
                            
                            prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                            df_pendientes['prob_late'] = prob_late
                            df_pendientes['late_checkout_pred'] = (prob_late > 0.5).astype(int)
                    except:
                        df_pendientes['late_checkout_pred'] = 0
                
                # Ordenar por prioridad
                if 'late_checkout_pred' in df_pendientes.columns:
                    df_pendientes = df_pendientes.sort_values(
                        by=['late_checkout_pred', 'habitacion_id'], 
                        ascending=[False, True]
                    )
                
                # Mostrar progreso
                st.subheader(f"📋 Progreso del día")
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
            
            # ===== SECCIÓN 3: HABITACIONES COMPLETADAS =====
            if completadas > 0:
                st.markdown("---")
                st.subheader(f"✅ Completadas ({completadas}/{total_asignadas})")
                
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
        
        # Añadir información de asignación
        if st.session_state.asignacion_global is not None:
            df['camarera_asignada'] = None
            for cam, habs in st.session_state.asignacion_global.items():
                df.loc[df['habitacion_id'].isin(habs), 'camarera_asignada'] = cam
        
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
