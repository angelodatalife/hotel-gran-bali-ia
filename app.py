# =============================================================================
# HOTEL GRAN BALI - SISTEMA IA DE GESTIÓN DE LIMPIEZA
# Versión con Stand By, contadores y separación Incidencias/Mantenimiento
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
    st.session_state.incidencias = []  # Problemas operativos
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
if 'standby' not in st.session_state:
    st.session_state.standby = []  # Habitaciones en espera (con problemas)

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
    """Asigna habitaciones por bloques de plantas adyacentes"""
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
        df_bloque = df_bloque.sort_values(
            by=['late_checkout_pred', 'habitacion_id'], 
            ascending=[False, True]
        )
        
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
# SIDEBAR - NAVEGACIÓN CON CONTADORES
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hotel.png", width=80)
    st.title("🏨 Hotel Gran Bali")
    st.markdown("---")
    
    # Contadores para el menú
    num_incidencias = len(st.session_state.incidencias)
    num_mantenimiento = len(st.session_state.mantenimiento)
    
    opciones_menu = {
        "📊 Gerente": "📊 Gerente",
        "🧹 Camarera": "🧹 Camarera",
        f"⚠️ Incidencias {f'({num_incidencias})' if num_incidencias > 0 else ''}": "⚠️ Incidencias",
        f"🔧 Mantenimiento {f'({num_mantenimiento})' if num_mantenimiento > 0 else ''}": "🔧 Mantenimiento",
        "📋 Dataset": "📋 Dataset"
    }
    
    seleccion = st.radio(
        "**Menú Principal**",
        list(opciones_menu.keys()),
        format_func=lambda x: x
    )
    pagina = opciones_menu[seleccion]
    
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
                    'asignacion_por_camarera', 'habitaciones_completadas', 'standby']:
            if key in st.session_state:
                if key in ['incidencias', 'mantenimiento', 'opiniones', 'habitaciones_completadas', 'standby']:
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
        
        sectores = {
            'Bajo (Pl 2-15)': {'plantas': list(range(2, 16)), 'camareras_base': 19},
            'Medio (Pl 16-30)': {'plantas': list(range(16, 31)), 'camareras_base': 11},
            'Alto (Pl 31-52)': {'plantas': list(range(31, 53)), 'camareras_base': 5}
        }
        
        datos_sectores = []
        for sector, info in sectores.items():
            df_sector = df[df['planta'].isin(info['plantas'])]
            num_hab = len(df_sector)
            
            cam_asignadas = 0
            if st.session_state.asignacion_por_camarera:
                for cam, df_cam in st.session_state.asignacion_por_camarera.items():
                    if any(p in info['plantas'] for p in df_cam['planta'].unique()):
                        cam_asignadas += 1
            
            if cam_asignadas == 0:
                cam_asignadas = info['camareras_base']
            
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
        
        col_graf_sect1, col_graf_sect2 = st.columns(2)
        with col_graf_sect1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Habitaciones', x=df_sectores['Sector'], y=df_sectores['Habitaciones'],
                                 marker_color=['#1E88E5', '#FFC107', '#DC143C']))
            fig.add_trace(go.Bar(name='Camareras', x=df_sectores['Sector'], y=df_sectores['Camareras'],
                                 marker_color=['#90CAF9', '#FFE082', '#FF8A80']))
            fig.update_layout(title='Habitaciones vs Camareras por Sector', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_graf_sect2:
            fig = px.bar(df_sectores, x='Sector', y='Tiempo/cam (min)', color='Sector',
                         title='Tiempo estimado por camarera (min)',
                         color_discrete_map={'Bajo (Pl 2-15)': '#1E88E5', 'Medio (Pl 16-30)': '#FFC107',
                                            'Alto (Pl 31-52)': '#DC143C'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle por Sector")
        st.dataframe(df_sectores, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("📈 Distribución por planta")
        
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            if 'planta' in df.columns:
                planta_counts = df['planta'].value_counts().sort_index()
                fig = px.bar(x=planta_counts.index, y=planta_counts.values,
                             labels={'x': 'Planta', 'y': 'Habitaciones'},
                             title='Habitaciones por planta')
                fig.add_vline(x=15.5, line_dash="dash", line_color="gray", annotation_text="Bajo/Medio")
                fig.add_vline(x=30.5, line_dash="dash", line_color="gray", annotation_text="Medio/Alto")
                st.plotly_chart(fig, use_container_width=True)
        
        with col_graf2:
            if 'prob_late' in df.columns:
                fig = px.histogram(df, x='prob_late', nbins=20,
                                  title='Distribución de probabilidades Late Checkout')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle de habitaciones")
        st.dataframe(df.head(100), use_container_width=True, height=400)

# =============================================================================
# VISTA CAMARERA (con Stand By)
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
                st.session_state.standby = []
                st.rerun()
        else:
            # Obtener asignación de esta camarera
            df_asignadas = st.session_state.asignacion_por_camarera.get(
                st.session_state.camarera_actual, 
                pd.DataFrame()
            )
            
            total_asignadas = len(df_asignadas)
            completadas = len(st.session_state.habitaciones_completadas)
            pendientes = total_asignadas - completadas - len(st.session_state.standby)
            
            # Mostrar información de la camarera
            col_info1, col_info2, col_info3 = st.columns([2, 2, 1])
            with col_info1:
                st.success(f"👤 {st.session_state.camarera_actual}")
            with col_info2:
                if len(df_asignadas) > 0:
                    plantas_unicas = sorted(df_asignadas['planta'].unique())
                    st.info(f"📌 Plantas: {min(plantas_unicas)}-{max(plantas_unicas)} ({len(plantas_unicas)} plantas)")
                else:
                    st.info("📌 Sin asignación")
            with col_info3:
                if st.button("🔄 Cambiar", key="btn_cambiar_usuario_principal"):
                    st.session_state.camarera_actual = None
                    st.session_state.habitaciones_completadas = []
                    st.session_state.standby = []
                    st.rerun()
            
            st.markdown("---")
            
            # ===== STAND BY (ARRIBA A LA DERECHA) =====
            if st.session_state.standby:
                with st.expander(f"⏸️ Stand By ({len(st.session_state.standby)} habitaciones)", expanded=True):
                    for idx, hab_id in enumerate(st.session_state.standby):
                        col_sb1, col_sb2, col_sb3 = st.columns([2, 1, 1])
                        with col_sb1:
                            st.markdown(f"⏸️ **Hab {hab_id}**")
                        with col_sb2:
                            st.caption("En espera")
                        with col_sb3:
                            if st.button("✅", key=f"sb_completar_{hab_id}_{idx}"):
                                st.session_state.habitaciones_completadas.append(hab_id)
                                st.session_state.standby.remove(hab_id)
                                st.rerun()
                        st.divider()
            
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
                    
                    # Reportar problema (sin detener cronómetro para mantenimiento)
                    with st.expander("⚠️ Reportar problema"):
                        tipo_reporte = st.selectbox(
                            "Tipo de problema",
                            ["Avería (Mantenimiento)", "Falta suministros", "Muy sucia", "Ocupada", "Otro"],
                            key="tipo_reporte_cron"
                        )
                        desc_reporte = st.text_area("Descripción", key="desc_reporte_cron")
                        
                        col_rep1, col_rep2 = st.columns(2)
                        with col_rep1:
                            if st.button("Reportar y continuar", key="btn_reporte_continuar", use_container_width=True):
                                hab_id = hab['habitacion_id']
                                
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
                                    st.success("🔧 Reporte enviado a Mantenimiento (puedes seguir limpiando)")
                                    st.rerun()
                                else:
                                    # Para estos casos, la habitación va a Stand By
                                    if hab_id not in st.session_state.standby:
                                        st.session_state.standby.append(hab_id)
                                    
                                    # Registrar incidencia
                                    st.session_state.incidencias.append({
                                        'habitacion': int(hab_id),
                                        'planta': int(hab['planta']),
                                        'tipo': tipo_reporte,
                                        'descripcion': desc_reporte,
                                        'timestamp': datetime.now().strftime("%H:%M"),
                                        'fecha': datetime.now().strftime("%d/%m/%Y"),
                                        'reportado_por': st.session_state.camarera_actual
                                    })
                                    st.warning(f"⏸️ Habitación {hab_id} movida a Stand By")
                                    
                                    # Terminar esta limpieza (la habitación no se completa)
                                    st.session_state.cronometro_activo = False
                                    st.session_state.habitacion_actual = None
                                    time.sleep(1)
                                    st.rerun()
                        
                        with col_rep2:
                            if st.button("Cancelar", key="btn_reporte_cancelar", use_container_width=True):
                                st.rerun()
                    
                    st.markdown("---")
            
            # ===== SECCIÓN 2: HABITACIONES PENDIENTES =====
            if total_asignadas > 0:
                # Excluir completadas y standby
                df_pendientes = df_asignadas[
                    ~df_asignadas['habitacion_id'].isin(st.session_state.habitaciones_completadas) &
                    ~df_asignadas['habitacion_id'].isin(st.session_state.standby)
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
                                if st.session_state.cronometro_activo:
                                    st.button("⏸️ En curso", key=f"btn_disabled_{row['habitacion_id']}_{i}",
                                             disabled=True, use_container_width=True)
                                else:
                                    if st.button("▶️ Iniciar", key=f"btn_iniciar_{row['habitacion_id']}_{i}",
                                                use_container_width=True):
                                        st.session_state.habitacion_actual = row
                                        st.session_state.cronometro_activo = True
                                        st.session_state.tiempo_inicio = datetime.now()
                                        st.rerun()
                            
                            st.divider()
            
            # ===== SECCIÓN 3: HABITACIONES COMPLETADAS =====
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
                            st.markdown("~~" + ("🏃 Late" if row.get('late_checkout_pred') == 1 else "🛏️ Normal") + "~~")
                        
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

elif pagina == "⚠️ Incidencias":
    st.title("⚠️ Panel de Incidencias Operativas")
    st.caption("Problemas: Falta suministros, Muy sucia, Ocupada, Otro")
    
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
