def asignar_habitaciones_por_modelos(df, num_cam, total_camaras=35):
    """
    Asigna habitaciones de forma EQUITATIVA usando los modelos entrenados:
    - XGBoost: calcula carga total y la distribuye
    - K-Means: agrupa plantas por cercanía
    - ANN: prioriza urgentes dentro de cada grupo
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    # 1. CALCULAR CARGA TOTAL DEL DÍA
    if 'tiempo_estimado' in df.columns:
        carga_total = df['tiempo_estimado'].sum()
        carga_por_cam = carga_total / total_camaras
        st.sidebar.info(f"📊 Carga total: {carga_total:.0f} min | Por camarera: {carga_por_cam:.0f} min")
    else:
        carga_por_cam = 120  # valor por defecto: 2 horas
    
    # 2. OBTENER CLUSTERS DE PLANTAS (K-Means)
    if modelos.get('kmeans') is not None:
        kmeans_model = modelos['kmeans']['modelo']
        scaler = modelos['kmeans']['scaler']
        features = modelos['kmeans']['features']
        
        # Crear perfil de todas las plantas
        plantas_unicas = sorted(df['planta'].unique())
        perfiles_plantas = []
        
        for planta in plantas_unicas:
            df_planta = df[df['planta'] == planta]
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
            perfiles_plantas.append(perfil)
        
        df_perfiles = pd.DataFrame(perfiles_plantas)
        X_perfiles = df_perfiles[features].values
        X_scaled = scaler.transform(X_perfiles)
        clusters = kmeans_model.predict(X_scaled)
        df_perfiles['cluster'] = clusters
        
        # Agrupar plantas por cluster
        plantas_por_cluster = {}
        for cluster in set(clusters):
            plantas_por_cluster[cluster] = df_perfiles[df_perfiles['cluster'] == cluster]['planta'].tolist()
    else:
        # Fallback: sectores tradicionales
        plantas_por_cluster = {
            0: list(range(2, 16)),   # Bajo
            1: list(range(16, 31)),  # Medio
            2: list(range(31, 53))   # Alto
        }
    
    # 3. ASIGNAR CAMARERAS A CLUSTERS (equitativo)
    clusters_list = list(plantas_por_cluster.keys())
    num_clusters = len(clusters_list)
    
    # Distribuir camareras proporcionalmente al tamaño de los clusters
    tamano_clusters = [len(plantas_por_cluster[c]) for c in clusters_list]
    total_plantas = sum(tamano_clusters)
    
    # Calcular cuántas camareras van a cada cluster
    camaras_por_cluster = {}
    asignadas = 0
    for i, cluster in enumerate(clusters_list):
        if i == num_clusters - 1:
            # Último cluster toma las restantes
            camaras_por_cluster[cluster] = total_camaras - asignadas
        else:
            prop = tamano_clusters[i] / total_plantas
            num = max(1, int(round(prop * total_camaras)))
            camaras_por_cluster[cluster] = num
            asignadas += num
    
    # Determinar qué cluster le toca a esta camarera
    cluster_actual = None
    acumulado = 0
    for cluster, num_camaras in camaras_por_cluster.items():
        if num_cam <= acumulado + num_camaras:
            cluster_actual = cluster
            break
        acumulado += num_camaras
    
    if cluster_actual is None:
        cluster_actual = clusters_list[-1]
    
    # 4. OBTENER PLANTAS DEL CLUSTER
    plantas_cluster = plantas_por_cluster[cluster_actual]
    
    # 5. FILTRAR HABITACIONES DEL CLUSTER
    df_candidatas = df[df['planta'].isin(plantas_cluster)].copy()
    
    # 6. APLICAR ANN PARA PRIORIZAR URGENTES
    if modelos.get('ann') is not None and 'late_checkout_pred' not in df_candidatas.columns:
        try:
            ann_model = modelos['ann']['modelo']
            scaler_ann = modelos['ann']['scaler']
            feature_cols = modelos['ann']['feature_cols']
            
            # Verificar que las columnas existen
            cols_disponibles = [c for c in feature_cols if c in df_candidatas.columns]
            if len(cols_disponibles) == len(feature_cols):
                X_ann = df_candidatas[feature_cols].values
                X_ann_scaled = scaler_ann.transform(X_ann)
                
                prob_late = ann_model.predict_proba(X_ann_scaled)[:, 1]
                df_candidatas['prob_late'] = prob_late
                df_candidatas['late_checkout_pred'] = (prob_late > 0.5).astype(int)
        except Exception as e:
            df_candidatas['late_checkout_pred'] = 0
    
    # 7. CALCULAR CARGA PARA ESTA CAMARERA (usando XGBoost implícitamente)
    if 'tiempo_estimado' in df_candidatas.columns:
        carga_cluster = df_candidatas['tiempo_estimado'].sum()
        num_camaras_cluster = camaras_por_cluster[cluster_actual]
        carga_objetivo = carga_cluster / num_camaras_cluster
        
        # Ordenar por prioridad y seleccionar hasta carga_objetivo
        if 'late_checkout_pred' in df_candidatas.columns:
            df_candidatas = df_candidatas.sort_values(
                by=['late_checkout_pred', 'tiempo_estimado'], 
                ascending=[False, False]
            )
        else:
            df_candidatas = df_candidatas.sort_values('tiempo_estimado', ascending=False)
        
        # Seleccionar habitaciones hasta alcanzar carga_objetivo
        carga_acumulada = 0
        indices_seleccionados = []
        for idx, row in df_candidatas.iterrows():
            if carga_acumulada + row['tiempo_estimado'] <= carga_objetivo * 1.2:  # tolerancia 20%
                indices_seleccionados.append(idx)
                carga_acumulada += row['tiempo_estimado']
            if len(indices_seleccionados) >= 10:  # máximo 10 habitaciones por camarera
                break
        
        df_final = df_candidatas.loc[indices_seleccionados].copy()
    else:
        # Sin tiempos, seleccionar máximo 10
        if 'late_checkout_pred' in df_candidatas.columns:
            df_candidatas = df_candidatas.sort_values('late_checkout_pred', ascending=False)
        df_final = df_candidatas.head(10).copy()
    
    return df_final
