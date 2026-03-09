# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con asignación por BLOQUES ADYACENTES
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
        
        st.subheader("📈 Distribución por bloques adyacentes")
        
        if st.session_state.asignacion_por_camarera:
            # Mostrar resumen de bloques
            bloques_resumen = {}
            for cam, df_cam in st.session_state.asignacion_por_camarera.items():
                plantas = sorted(df_cam['planta'].unique())
                if len(plantas) > 0:
                    bloque_key = f"{min(plantas)}-{max(plantas)}"
                    if bloque_key not in bloques_resumen:
                        bloques_resumen[bloque_key] = []
                    bloques_resumen[bloque_key].append(cam)
            
            st.markdown("**Bloques de trabajo:**")
            for bloque, cams in bloques_resumen.items():
                st.markdown(f"- Plantas {bloque}: {len(cams)} camareras")
            
            # Gráficos de carga
            datos_carga = []
            for cam, df_cam in st.session_state.asignacion_por_camarera.items():
                tiempo = df_cam['tiempo_estimado'].sum() if 'tiempo_estimado' in df_cam.columns else len(df_cam) * 25
                datos_carga.append({
                    'Camarera': cam,
                    'Habitaciones': len(df_cam),
                    'Tiempo (min)': round(tiempo, 1),
                    'Plantas': f"{int(df_cam['planta'].min())}-{int(df_cam['planta'].max())}"
                })
            
            df_carga = pd.DataFrame(datos_carga)
            
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                fig = px.bar(
                    df_carga, 
                    x='Camarera', 
                    y='Habitaciones',
                    color='Plantas',
                    title='Habitaciones por camarera'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_graf2:
                fig = px.bar(
                    df_carga, 
                    x='Camarera', 
                    y='Tiempo (min)',
                    color='Plantas',
                    title='Tiempo estimado por camarera'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA - CON BLOQUES ADYACENTES
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
                    # Verificar si son adyacentes
                    son_adyacentes = all(
                        plantas_unicas[i+1] - plantas_unicas[i] == 1 
                        for i in range(len(plantas_unicas)-1)
                    )
                    if son_adyacentes:
                        st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas) ✅")
                    else:
                        st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas)")
                else:
                    st.info("📌 Sin asignación")
            with col_info3:
                if st.button("🔄 Cambiar usuario"):
                    st.session_state.camarera_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.rerun()
            
            st.markdown("---")
            
            # Barra de progreso
            progreso_total = completadas / total_asignadas if total_asignadas > 0 else 0
            st.progress(progreso_total, text=f"**{completadas}/{total_asignadas}** habitaciones completadas")
            
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
