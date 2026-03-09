# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión para Streamlit Cloud
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import os
import time

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
    for nombre, archivo in archivos_modelos.items():
        if os.path.exists(archivo):
            try:
                with open(archivo, 'rb') as f:
                    modelos[nombre] = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ No se pudo cargar {nombre}: {str(e)}")
                modelos[nombre] = None
        else:
            st.warning(f"⚠️ Archivo no encontrado: {archivo}")
            modelos[nombre] = None
    
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
                    'cronometro_activo', 'tiempo_inicio', 'habitacion_actual']:
            if key in st.session_state:
                if key == 'incidencias':
                    st.session_state[key] = []
                elif key == 'opiniones':
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
                checkouts = len(df[df['clase_checkout'] == 'Salida']) if 'clase_checkout' in df.columns else 0
                st.metric("Checkouts", checkouts)
            with col_metric3:
                repasos = len(df[df['clase_checkout'] == 'Repaso']) if 'clase_checkout' in df.columns else 0
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
            st.subheader("📊 Tipo de servicio")
            if 'clase_checkout' in df.columns:
                tipo_counts = df['clase_checkout'].value_counts()
                fig = px.pie(
                    values=tipo_counts.values,
                    names=tipo_counts.index,
                    title='Salidas vs Repasos'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA - CON LIMPIEZA EN CURSO ARRIBA
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
                # Inicializar lista de completadas al seleccionar camarera
                st.session_state.habitaciones_completadas = []
                st.rerun()
        else:
            # Inicializar lista de completadas si no existe
            if 'habitaciones_completadas' not in st.session_state:
                st.session_state.habitaciones_completadas = []
            
            # Obtener número de camarera
            num_cam = int(st.session_state.camarera_actual.split()[1])
            
            # DEFINIR CLÚSTERES DE PLANTAS (compactos, adyacentes)
            clusteres_plantas = {
                # Sector Bajo (19 camareras) - plantas 2-15
                1: [2, 3, 4],      # Camareras 1-3
                2: [5, 6],         # Camareras 4-5
                3: [7, 8, 9],      # Camareras 6-8
                4: [10, 11],       # Camareras 9-10
                5: [12, 13],       # Camareras 11-12
                6: [14, 15],       # Camareras 13-14
                
                # Sector Medio (11 camareras) - plantas 16-30
                7: [16, 17, 18],   # Camareras 15-17
                8: [19, 20, 21],   # Camareras 18-20
                9: [22, 23, 24],   # Camareras 21-23
                10: [25, 26],      # Camareras 24-25
                11: [27, 28, 29, 30], # Camareras 26-29
                
                # Sector Alto (5 camareras) - plantas 31-52
                12: [31, 32, 33, 34, 35],  # Camarera 30
                13: [36, 37, 38, 39, 40],  # Camarera 31
                14: [41, 42, 43, 44, 45],  # Camarera 32
                15: [46, 47, 48, 49, 50],  # Camarera 33
                16: [51, 52],               # Camarera 34
            }
            
            # Asignar clúster según número de camarera
            if num_cam <= 3:
                cluster = 1
            elif num_cam <= 5:
                cluster = 2
            elif num_cam <= 8:
                cluster = 3
            elif num_cam <= 10:
                cluster = 4
            elif num_cam <= 12:
                cluster = 5
            elif num_cam <= 14:
                cluster = 6
            elif num_cam <= 17:
                cluster = 7
            elif num_cam <= 20:
                cluster = 8
            elif num_cam <= 23:
                cluster = 9
            elif num_cam <= 25:
                cluster = 10
            elif num_cam <= 29:
                cluster = 11
            elif num_cam == 30:
                cluster = 12
            elif num_cam == 31:
                cluster = 13
            elif num_cam == 32:
                cluster = 14
            elif num_cam == 33:
                cluster = 15
            else:
                cluster = 16
            
            plantas_asignadas = clusteres_plantas[cluster]
            sector = 'Bajo' if cluster <= 6 else ('Medio' if cluster <= 11 else 'Alto')
            
            # Mostrar información de la camarera
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                st.info(f"📌 Plantas: {min(plantas_asignadas)}-{max(plantas_asignadas)} ({sector})")
            with col_info3:
                if st.button("🔄 Cambiar usuario"):
                    st.session_state.camarera_actual = None
                    if 'habitaciones_completadas' in st.session_state:
                        del st.session_state.habitaciones_completadas
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
                    
                    # Reportar incidencia (dentro del mismo contenedor)
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
                            
                            # También mover a completadas aunque haya incidencia
                            hab_id = hab['habitacion_id']
                            st.session_state.habitaciones_completadas.append(hab_id)
                            
                            st.session_state.cronometro_activo = False
                            st.session_state.habitacion_actual = None
                            time.sleep(1)
                            st.rerun()
                    
                    st.markdown("---")
            
            # ===== SECCIÓN 2: HABITACIONES PENDIENTES =====
            # Filtrar habitaciones de sus plantas asignadas que NO estén completadas
            df_asignadas = df[df['planta'].isin(plantas_asignadas)].copy()
            
            # Excluir las que ya están completadas
            if st.session_state.habitaciones_completadas:
                df_asignadas = df_asignadas[~df_asignadas['habitacion_id'].isin(
                    st.session_state.habitaciones_completadas
                )]
            
            # Si no hay suficientes en sus plantas, tomar las más cercanas
            if len(df_asignadas) < 4:
                todas_plantas = sorted(set(df['planta']))
                idx_actual = todas_plantas.index(min(plantas_asignadas))
                
                plantas_extra = []
                if idx_actual > 0:
                    plantas_extra.append(todas_plantas[idx_actual - 1])
                if idx_actual + len(plantas_asignadas) < len(todas_plantas):
                    plantas_extra.append(todas_plantas[idx_actual + len(plantas_asignadas)])
                
                df_extra = df[df['planta'].isin(plantas_extra)]
                # Excluir completadas también de las extra
                if st.session_state.habitaciones_completadas:
                    df_extra = df_extra[~df_extra['habitacion_id'].isin(
                        st.session_state.habitaciones_completadas
                    )]
                df_asignadas = pd.concat([df_asignadas, df_extra]).drop_duplicates()
            
            # Limitar a máximo 8 habitaciones pendientes
            df_pendientes = df_asignadas.copy()
            if len(df_pendientes) > 8:
                if 'clase_checkout' in df_pendientes.columns:
                    urgentes = df_pendientes[df_pendientes['clase_checkout'] == 'Salida']
                    no_urgentes = df_pendientes[df_pendientes['clase_checkout'] != 'Salida']
                    
                    urgentes = urgentes.head(4)
                    restantes = 8 - len(urgentes)
                    no_urgentes = no_urgentes.head(restantes)
                    df_pendientes = pd.concat([urgentes, no_urgentes])
                else:
                    df_pendientes = df_pendientes.head(8)
            
            # Ordenar pendientes: urgentes primero, luego por número descendente
            if 'clase_checkout' in df_pendientes.columns:
                df_pendientes['es_urgente'] = (df_pendientes['clase_checkout'] == 'Salida').astype(int)
                df_pendientes = df_pendientes.sort_values(
                    by=['es_urgente', 'habitacion_id'], 
                    ascending=[False, False]
                ).drop('es_urgente', axis=1)
            else:
                df_pendientes = df_pendientes.sort_values('habitacion_id', ascending=False)
            
            st.subheader(f"📋 Pendientes ({len(df_pendientes)} restantes)")
            
            if len(df_pendientes) == 0:
                st.success("🎉 ¡Has completado todas tus habitaciones!")
                st.balloons()
            else:
                for idx, row in df_pendientes.iterrows():
                    with st.container():
                        cols = st.columns([3, 2, 2, 3])
                        
                        with cols[0]:
                            tipo_emoji = "🔴" if row.get('clase_checkout') == 'Salida' else "🟡"
                            st.markdown(f"{tipo_emoji} **Hab {int(row['habitacion_id'])}**")
                            st.caption(f"Planta {int(row['planta'])}")
                        
                        with cols[1]:
                            tipo_serv = row.get('clase_checkout', 'N/A')
                            if tipo_serv == 'Salida':
                                st.markdown("🏃 **Checkout**")
                            else:
                                st.markdown("🛏️ **Repaso**")
                        
                        with cols[2]:
                            if 'tiempo_estimado' in row:
                                st.markdown(f"⏱️ **{row['tiempo_estimado']} min**")
                        
                        with cols[3]:
                            # Deshabilitar botón si hay una limpieza en curso
                            disabled = st.session_state.cronometro_activo
                            if st.button(
                                f"▶️ Iniciar", 
                                key=f"btn_{idx}", 
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
                
                # Obtener datos de las completadas
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
                            tipo_serv = row.get('clase_checkout', 'N/A')
                            st.markdown(f"~~{tipo_serv}~~")
                        
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
                        if st.button("✓ Resolver", key=f"resolver_{len(st.session_state.incidencias)}"):
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
            st.metric("Con opiniones", len([o for o in st.session_state.opiniones]))
        with col_met3:
            st.metric("Incidencias", len(st.session_state.incidencias))
        with col_met4:
            if 'clase_checkout' in df.columns:
                checkouts = len(df[df['clase_checkout'] == 'Salida'])
                st.metric("Checkouts", checkouts)
        
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
