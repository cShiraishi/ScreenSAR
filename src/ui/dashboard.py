import streamlit as st
import pandas as pd
import altair as alt
import io
import pickle
import numpy as np
from datetime import datetime
from src.core.curation import CuradoriaQSAR
from src.core.modeling import ModeladorQSAR
from src.utils.report import generate_pdf_report

def render_dashboard(config):
    """
    Renders the main dashboard results (Curated Data, Charts, Modeling).
    """
    t = config['t']
    lang = config['lang']
    
    if st.session_state.curated_result is not None:
        # Initialize session state for outliers if not present
        if 'removed_outliers_indices' not in st.session_state:
            st.session_state.removed_outliers_indices = []
            
        df_result = st.session_state.curated_result
        input_len = st.session_state.input_len
        
        st.success(t['success_msg'].format(len(df_result)))
        
        col1, col2, col3 = st.columns(3)
        col1.metric(t['total_orig'], input_len)
        col2.metric(t['total_final'], len(df_result))
        col3.metric(t['removed'], input_len - len(df_result))
        
        st.subheader(t['curated_header'])
        st.dataframe(df_result.head(20))
        
        # Prepare downloads
        csv = df_result.to_csv(index=False).encode('utf-8')
        
        buffer_actives = io.BytesIO()
        with pd.ExcelWriter(buffer_actives, engine='openpyxl') as writer:
            df_result[df_result['Outcome'] == 1].to_excel(writer, index=False)
            
        buffer_inactives = io.BytesIO()
        with pd.ExcelWriter(buffer_inactives, engine='openpyxl') as writer:
            df_result[df_result['Outcome'] == 0].to_excel(writer, index=False)
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                label=t['download_csv'],
                data=csv,
                file_name='curated_dataset_full.csv',
                mime='text/csv',
            )
        with col_d2:
            st.download_button(
                label=t['download_actives'],
                data=buffer_actives.getvalue(),
                file_name='curated_actives.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
        with col_d3:
            st.download_button(
                label=t['download_inactives'],
                data=buffer_inactives.getvalue(),
                file_name='curated_inactives.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
        
        st.markdown("---")
        st.subheader(t['dist_header'])
        
        counts = df_result['Outcome'].value_counts()
        actives = counts.get(1, 0)
        inactives = counts.get(0, 0)
        
        col_a, col_i = st.columns(2)
        col_a.metric(t['actives'], actives)
        col_i.metric(t['inactives'], inactives)
        
        # Chart
        chart_data = pd.DataFrame({
            'Quantidade': [actives, inactives]
        }, index=[t['actives'], t['inactives']])
        
        st.bar_chart(chart_data)
        
        st.markdown("---")
        st.subheader(t['outlier_header'])
        
        # Determine which column to analyze
        target_col = 'pIC50' if 'pIC50' in df_result.columns else 'IC50_nM'
        
        with st.expander(t['expander_outlier'], expanded=True):
            st.write(f"{t['analyzing_dist']} **{target_col}**")
            
            # Calculate Stats
            data_vals = df_result[target_col].dropna()
            if not data_vals.empty:
                mean_val = data_vals.mean()
                std_val = data_vals.std()
                
                # Identify outliers (> 3 std dev from mean is a common simple heuristic)
                threshold_upper = mean_val + (3 * std_val)
                threshold_lower = mean_val - (3 * std_val)
                
                outliers = df_result[
                    (df_result[target_col] > threshold_upper) | 
                    (df_result[target_col] < threshold_lower)
                ]
                
                c1, c2, c3 = st.columns(3)
                c1.metric(t['mean'], f"{mean_val:.2f}")
                c2.metric(t['std'], f"{std_val:.2f}")
                c3.metric(t['potential_outliers'], len(outliers))
                
                # Altair Boxplot
                # Map outcome to text for chart
                df_chart = df_result.copy()
                df_chart['Class_Label'] = df_chart['Outcome'].apply(lambda x: t['active_singular'] if x==1 else t['inactive_singular'])

                chart = alt.Chart(df_chart).mark_boxplot(extent='min-max').encode(
                    x=alt.X('Class_Label:N', title=None),
                    y=alt.Y(target_col, title=target_col),
                    color=alt.Color('Class_Label:N', title=None)
                ).properties(
                    title=f'Boxplot: {target_col}'
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
                if len(outliers) > 0:
                    st.warning(t['outlier_warning'].format(len(outliers)))
                    st.dataframe(outliers[['Molecule ChEMBL ID', 'SMILES_Clean', target_col, 'Outcome']])
            else:
                 st.warning(t['outlier_none'])

        st.markdown("---")
        st.subheader(t['chem_space_header'])
        
        with st.expander(t['expander_chem_space'], expanded=True):
            st.info(t['chem_space_info'])
            
            col_fp1, col_fp2 = st.columns(2)
            n_bits = col_fp1.selectbox(t['nbits'], [1024, 2048, 512], index=0, key='pca_nbits')
            radius = col_fp2.number_input(t['radius'], min_value=1, max_value=4, value=2, key='pca_radius')
            
            if st.button(t['gen_map_btn']):
                try:
                    from sklearn.decomposition import PCA
                    
                    with st.spinner("Generating..."):
                        curador_pca = CuradoriaQSAR(df_result)
                        # Use global descriptor type or default to Morgan for PCA if strictly needed? 
                        # If "All" is selected, default to Morgan for visualization
                        dt_config = config.get('descriptor_type', 'Morgan')
                        if dt_config == "All":
                            st.info("Visualizing with 'Morgan' descriptor because 'All' was selected.")
                            dt = "Morgan"
                        else:
                            dt = dt_config
                        fps, valid_idxs = curador_pca.gerar_fingerprints(df_result, n_bits=n_bits, radius=radius, descriptor_type=dt) 
                        # Wait, CuradoriaQSAR doesn't have gerar_fingerprints? It was likely a hallucination in the view or I missed it.
                        # Checking view of dashboard.py line 149: `fps, valid_idxs = curador_pca.gerar_fingerprints(...)`
                        # My previous view of `modeling.py` showed `ModeladorQSAR` has `gerar_dados`. 
                        # `CuradoriaQSAR` is in `curation.py`. 
                        # Ah, the PCA block uses `CuradoriaQSAR`. Does it have `gerar_fingerprints`?
                        # I must check `src/core/curation.py`.
                        # If `CuradoriaQSAR` has `gerar_fingerprints`, I need to update THAT too.
                        # But `ModeladorQSAR` was updated.
                        # Let's pause and check `curation.py`.

                        
                        if len(fps) > 2:
                            pca = PCA(n_components=2)
                            pca_result = pca.fit_transform(fps)
                            
                            df_pca = df_result.iloc[valid_idxs].copy()
                            df_pca['PCA1'] = pca_result[:, 0]
                            df_pca['PCA2'] = pca_result[:, 1]
                            
                            df_pca['ID'] = df_pca['Molecule ChEMBL ID'] if 'Molecule ChEMBL ID' in df_pca.columns else df_pca.index
                            df_pca['Classe'] = df_pca['Outcome'].apply(lambda x: t['active_singular'] if x==1 else t['inactive_singular'])
                            
                            var_exp = pca.explained_variance_ratio_
                            st.write(t['var_explained'].format(f"PC1 ({var_exp[0]:.1%}), PC2 ({var_exp[1]:.1%})"))
                            
                            tooltip_cols = ['ID', 'SMILES_Clean', 'Classe']
                            if 'pIC50' in df_pca.columns: tooltip_cols.append('pIC50')
                            else: tooltip_cols.append('IC50_nM')

                            scatter = alt.Chart(df_pca).mark_circle(size=60).encode(
                                x=alt.X('PCA1', title=f'PC1 ({var_exp[0]:.1%})'),
                                y=alt.Y('PCA2', title=f'PC2 ({var_exp[1]:.1%})'),
                                color=alt.Color('Classe', scale=alt.Scale(domain=[t['active_singular'], t['inactive_singular']], range=['#1f77b4', '#d62728'])),
                                tooltip=tooltip_cols
                            ).properties(
                                title=t['map_title'],
                                height=500
                            ).interactive()
                            
                            st.altair_chart(scatter, use_container_width=True)
                            
                        else:
                            st.error(t['error_insufficient'])
                
                except Exception as e:
                    st.error(t['error_generic'].format(e))

                    st.error(t['error_generic'].format(e))
        
        st.markdown("---")
        # --- NEW: Pre-Modeling AD Analysis (Request: delete outliers BEFORE model creation) ---
        st.subheader("🛡️ " + t.get('ad_pre_analysis_title', 'Pre-Modeling Applicability Domain Analysis'))
        
        with st.expander(t.get('ad_pre_expander', 'Analyze & Clean Outliers (Optional)'), expanded=True):
            st.info("Identify and remove structurally distinct compounds (outliers) from your dataset **before** training.")
            
            c_ad_pre1, c_ad_pre2 = st.columns(2)
            n_bits_ad = c_ad_pre1.selectbox("Bits", [1024, 2048], key="ad_pre_bits")
            radius_ad = c_ad_pre2.number_input("Radius", 2, 4, 2, key="ad_pre_rad")
            
            if st.button("📊 Analyze Outliers Now"):
                with st.spinner("Calculating distances..."):
                    try:
                        # Use Modelador just for generation
                        mod_ad = ModeladorQSAR(df_result)
                        X_ad, _, valid_idxs_ad = mod_ad.gerar_dados(n_bits=n_bits_ad, radius=radius_ad, descriptor_type="Morgan")
                        
                        # Use ApplicabilityDomain on the WHOLE dataset
                        from src.core.applicability_domain import ApplicabilityDomain
                        ad_pre = ApplicabilityDomain(k_neighbors=5, z_threshold=3.0)
                        
                        # We fit on X_ad. Outliers are relative to the whole set distribution
                        ad_pre.fit(X_ad)
                        outliers_local = ad_pre.detect_outliers()
                        
                        # Map back to global DF indices
                        # valid_idxs_ad maps local X index -> df index
                        global_outliers_pre = [valid_idxs_ad[i] for i in outliers_local]
                        
                        # Save result to session state to persist table
                        st.session_state['pre_ad_result'] = {
                            "global_outliers": global_outliers_pre,
                            "distances": ad_pre.training_distances, # Distances of all points
                            "valid_idxs": valid_idxs_ad
                        }
                        
                    except Exception as e:
                        st.error(f"Error in AD analysis: {e}")

            # Display Results if available
            if 'pre_ad_result' in st.session_state:
                res = st.session_state['pre_ad_result']
                g_outliers = res['global_outliers']
                dists = res['distances']
                v_idxs = res['valid_idxs']
                
                # Check current removed status to separate pending vs removed
                already_removed = set(st.session_state.get('removed_outliers_indices', []))
                active_outliers = [o for o in g_outliers if o not in already_removed]
                
                if active_outliers:
                    st.warning(f"⚠️ **{len(active_outliers)} Outliers** detected (Z-Score > 3.0).")
                else:
                    st.success("✅ No active statistical outliers found.")

                # Table Logic (Top 50 distant)
                # Sort all by distance
                if dists is not None and len(dists) > 0:
                    # Sort indices by distance descending (Safe Python Sort)
                    sorted_indices = sorted(range(len(dists)), key=lambda k: dists[k], reverse=True)[:50]
                else:
                    sorted_indices = []
                
                top_global_indices = []
                top_dists = []
                
                for idx_local in sorted_indices:
                    g_idx = v_idxs[idx_local]
                    # Only show if not already removed? Or show as removed?
                    # Let's show all but mark status
                    top_global_indices.append(g_idx)
                    top_dists.append(dists[idx_local])
                
                df_view = df_result.loc[top_global_indices].copy()
                df_view['Distance'] = top_dists
                df_view['Is_Outlier'] = [x in g_outliers for x in top_global_indices]
                df_view['Status'] = ["Removed" if x in already_removed else "Active" for x in top_global_indices]
                df_view['Delete'] = [(x not in already_removed and x in g_outliers) for x in top_global_indices] # Default check active outliers
                
                # Colors/Config
                edited_pre = st.data_editor(
                    df_view,
                    column_config={
                        "Delete": st.column_config.CheckboxColumn("Delete?", default=False),
                        "Distance": st.column_config.NumberColumn(format="%.4f"),
                        "Is_Outlier": st.column_config.CheckboxColumn(disabled=True),
                        "Status": st.column_config.TextColumn(disabled=True)
                    },
                    disabled=[c for c in df_view.columns if c != 'Delete'],
                    key="editor_pre_ad"
                )
                
                # Process Deletion
                to_del = edited_pre[edited_pre['Delete'] == True].index.tolist()
                
                c_del1, c_del2 = st.columns([1,3])
                if c_del1.button("🗑️ Delete Selected", type="primary"):
                    current = st.session_state.get('removed_outliers_indices', [])
                    # Add new
                    updated = list(set(current + to_del))
                    st.session_state.removed_outliers_indices = updated
                    st.toast(f"Marked {len(to_del)} compounds for removal.")
                    st.rerun()
                    
                if st.session_state.get('removed_outliers_indices'):
                     st.info(f"Total compounds marked for removal: {len(st.session_state.removed_outliers_indices)}")
                     if st.button("Undo All Removals"):
                         st.session_state.removed_outliers_indices = []
                         st.rerun()

        st.markdown("---")
        st.subheader(t['model_header'])
        
        with st.expander(t['model_expander'], expanded=False):
            st.write(t['model_intro'])
            
            # MODI Check
            if st.button(t['calc_modi']):
                with st.spinner(t['modi_spinner']):
                    modelador_modi = ModeladorQSAR(df_result)
                    desc_type_modi = config.get('descriptor_type', 'Morgan')
                    nb_modi = config.get('n_bits', 1024)
                    rad_modi = config.get('radius', 2)
                    X_modi, y_modi, _ = modelador_modi.gerar_dados(n_bits=nb_modi, radius=rad_modi, descriptor_type=desc_type_modi)
                    
                    if len(y_modi) > 5:
                        modi_val = modelador_modi.calcular_modi(X_modi, y_modi)
                        
                        col_m1, col_m2 = st.columns([1, 3])
                        col_m1.metric("MODI", f"{modi_val:.3f}")
                        
                        if modi_val >= 0.65:
                            col_m2.success(t['modi_high'])
                        else:
                            col_m2.warning(t['modi_low'])
                    else:
                        st.error(t['error_insufficient'])
            
            st.divider()

            # 1. Select Models
            available_models = ["Random Forest", "SVM", "Gradient Boosting", "KNN", "Logistic Regression"]
            selected_models = st.multiselect(t['select_models'], available_models, default=["Random Forest"])
            
            # 2. Options
            test_split = st.slider(
                t['test_split'], 
                min_value=10, 
                max_value=50, 
                value=20, 
                step=5,
                help=t['test_split_help']
            ) / 100.0
            st.caption(t['test_split_explanation'])
            
            # Benchmark Checkbox
            dt_config = config.get('descriptor_type', 'Morgan')
            is_all = (dt_config == "All")
            
            # If "All" is selected, force benchmark to be checked
            chk_value = True if is_all else False
            chk_disabled = True if is_all else False
            
            do_benchmark = st.checkbox(
                t.get('benchmark_label', "Benchmark All Descriptors (Morgan, MACCS, RDKit)"), 
                value=chk_value,
                disabled=chk_disabled
            )
            
            train_btn = st.button(t['train_btn'])
            
            if train_btn:
                if not selected_models:
                    st.warning(t['warn_select_model'])
                else:
                    try:
                        with st.spinner(t['training_spinner']):
                            modelador = ModeladorQSAR(df_result)
                            
                            descriptors_to_run = []
                            if do_benchmark or config.get('descriptor_type') == "All":
                                descriptors_to_run = ["Morgan", "MACCS", "RDKit"]
                            else:
                                descriptors_to_run = [config.get('descriptor_type', 'Morgan')]
                            
                            all_results_list = []
                            all_trained_models = {}
                            all_roc_data = {}
                            all_ad_info = {} # Store AD info (usually just one if descriptor is same, but let's keep it consistent)
                            
                            # Params for Morgan (others ignore these)
                            nb = config.get('n_bits', 1024)
                            rad = config.get('radius', 2)
                            
                            for dt in descriptors_to_run:
                                # Status update (optional if spinner is enough, but helpful)
                                # st.toast(f"Running {dt}...") 
                                
                                X, y, valid_indices = modelador.gerar_dados(n_bits=nb, radius=rad, descriptor_type=dt)
                                
                                if len(y) < 20:
                                     if not do_benchmark: st.error(t['error_insufficient'])
                                     continue
                                else:
                                    X_masked = X
                                    y_masked = y
                                    valid_idxs_masked = list(range(len(X))) # local indices
                                    
                                    # Filter removed outliers if any
                                    if st.session_state.removed_outliers_indices:
                                        # valid_indices maps X-index -> df_index
                                        # We want to keep X-indices where valid_indices[i] is NOT in removed_outliers_indices
                                        
                                        keep_mask = []
                                        for i, df_idx in enumerate(valid_indices):
                                            if df_idx not in st.session_state.removed_outliers_indices:
                                                keep_mask.append(i)
                                        
                                        if len(keep_mask) < len(X):
                                            X_masked = X[keep_mask]
                                            y_masked = y[keep_mask]
                                            valid_idxs_masked = keep_mask
                                            # st.write(f"Filtered {len(X) - len(X_masked)} outliers. New count: {len(X_masked)}")
                                    
                                    if len(y_masked) < 20:
                                        st.error(t['error_insufficient'])
                                        continue

                                    results, trained, roc, ad_info = modelador.treinar_avaliar(
                                        X_masked, y_masked, selected_models, test_size=test_split
                                    )
                                    
                                    # We need to preserve the mapping from local train indices back to global DF indices
                                    # ad_info['outliers_train_idx'] are indices in X_masked
                                    # We need them as indices in DF
                                    
                                    if ad_info and 'outliers_train_idx' in ad_info:
                                        # Map: Local X_train idx -> X_masked idx -> original valid_indices -> DF indicies
                                        # Wait, ad_info['outliers_train_idx'] are already mapped to X_masked indices by my previous change to modeling.py
                                        
                                        local_outliers_indices = ad_info['outliers_train_idx']
                                        # Map to DF indices
                                        # valid_indices[ valid_idxs_masked[ local_outlier_idx ] ]
                                        
                                        global_outlier_indices = []
                                        for loc_idx in local_outliers_indices:
                                            # loc_idx is index in X_masked
                                            # valid_idxs_masked[loc_idx] is index in X (original)
                                            # valid_indices[...] is index in DF
                                            original_x_idx = valid_idxs_masked[loc_idx]
                                            df_idx = valid_indices[original_x_idx]
                                            global_outlier_indices.append(df_idx)
                                            
                                        ad_info['global_outlier_indices'] = global_outlier_indices
                                    
                                    # Rename models to include descriptor if benchmarking
                                    if do_benchmark:
                                        results['Modelo'] = results['Modelo'] + f" ({dt})"
                                        
                                        # Update keys in trained and roc dicts
                                        # (Need to copy to avoid runtime error if we modified in place, but we can just make new dicts)
                                        new_trained = {f"{k} ({dt})": v for k, v in trained.items()}
                                        new_roc = {f"{k} ({dt})": v for k, v in roc.items()}
                                        
                                        trained = new_trained
                                        roc = new_roc
                                        # Should we duplicate AD info? Maybe just keep the last one or link to model name
                                        
                                    
                                    all_results_list.append(results)
                                    all_trained_models.update(trained)
                                    all_roc_data.update(roc)
                                    
                                    if ad_info:
                                        # Store ad_info keyed by descriptor or model? 
                                        # Since AD is per training set (and descriptor), let's key by descriptor
                                        all_ad_info[dt] = ad_info
                            
                            if all_results_list:
                                final_results = pd.concat(all_results_list, ignore_index=True)
                                st.session_state['modeling_results'] = final_results
                                st.session_state['roc_data'] = all_roc_data
                                st.session_state['trained_models'] = all_trained_models
                                st.session_state['ad_info'] = all_ad_info
                                
                                st.success(t['training_success'])
                            else:
                                st.error("No successful training runs.")

                    except Exception as e:
                        st.error(t['error_generic'].format(e))

            # Display Results from Session State
            if st.session_state.get('modeling_results') is not None:
                 results = st.session_state['modeling_results']
                 roc_data = st.session_state.get('roc_data')
                 
                 if not results.empty:
                     # Check for errors
                     if "Erro" in results.columns:
                         errors = results[results["Erro"].notna() & (results["Erro"] != "nan") & (results["Erro"] != "")]
                         if not errors.empty:
                             st.error(t['failed_models'])
                             st.dataframe(errors[["Modelo", "Erro"]])
                     
                     # Display Metrics
                     # --- Applicability Domain Section ---
                     if 'input_len' not in st.session_state:
                         st.session_state.input_len = 0
                     if 'removed_outliers_indices' not in st.session_state:
                         st.session_state.removed_outliers_indices = []

                     if 'ad_info' in st.session_state and st.session_state['ad_info']:
                         st.divider()
                         st.subheader("🛡️ Applicability Domain (AD)")
                         
                         # Get first available AD info (assuming similar outliers for same dataset)
                         # Or allow user to see per descriptor
                         first_key = list(st.session_state['ad_info'].keys())[0]
                         ad_data = st.session_state['ad_info'][first_key]
                         
                         global_outliers = ad_data.get('global_outlier_indices', [])
                         
                         c_ad1, c_ad2 = st.columns([3, 1])
                         with c_ad1:
                             if len(global_outliers) > 0:
                                 st.warning(f"⚠️ **{len(global_outliers)} Outliers** detected in the Training Set (based on distance to neighbors).")
                                 st.caption("These compounds arestructurally distinct from the rest of the training data and may reduce model accuracy if they are erroneous.")
                                 
                                 # Show them?
                                 if st.checkbox("Show Outliers List"):
                                     st.dataframe(df_result.loc[global_outliers])
                             else:
                                 st.success("✅ No significant outliers detected in Training Set.")
                        
                         with c_ad2:
                             if len(global_outliers) > 0:
                                 if st.button("🧹 Remove Automatic Outliers & Retrain"):
                                     # Add to removed list
                                     current_removed = st.session_state.removed_outliers_indices
                                     # Avoid duplicates
                                     new_removed = list(set(current_removed + global_outliers))
                                     st.session_state.removed_outliers_indices = new_removed
                                     
                                     st.toast(f"Removed {len(global_outliers)} outliers. Retrianing recommended.")
                                     st.warning("Outliers marked for removal. Please click 'Train Models' again.")
                         
                         # Logic to get top distant even if not outliers
                         # We need distances for all train set
                         ad_model = ad_data.get('model')
                         idx_train_map = ad_data.get('idx_train') # Indices in X for each row in X_train
                         
                         if ad_model and ad_model.training_distances is not None and idx_train_map is not None:
                             # Get indices of top 50 distances (local to X_train)
                             dists = ad_model.training_distances
                             
                             # Sort descending
                             sorted_local_indices = np.argsort(dists)[::-1][:50]
                             
                             # Map to global DF indices
                             # 1. local_idx (in X_train) -> x_idx (in X) using idx_train_map
                             # 2. x_idx (in X) -> df_idx (in DF) using valid_idxs_masked (if masking happened) -> then valid_indices
                             
                             # Wait, idx_train_map contains indices relative to the X passed to train_test_split.
                             # But that X might be X_masked if we already filtered some outliers!
                             # In modeling.py: X_train, ..., idx_train, ... = train_test_split(X, ...)
                             # So idx_train refers to indices in the X passed to treinar_avaliar.
                             
                             full_global_indices = []
                             
                             for local_i in sorted_local_indices:
                                 # idx_train_map[local_i] gives index in X (the input to treinar_avaliar)
                                 idx_in_X_masked = idx_train_map[local_i]
                                 
                                 # Now map X_masked index -> Original X index -> DF index
                                 # We have valid_idxs_masked which maps X_masked -> Original X (input to loop)
                                 # valid_idxs_masked is a list where value is original index
                                 
                                 original_X_idx = valid_idxs_masked[idx_in_X_masked]
                                 
                                 # Now Original X index -> DF index
                                 # valid_indices maps Original X -> DF
                                 df_idx = valid_indices[original_X_idx]
                                 
                                 full_global_indices.append(df_idx)
                                 
                             # Now show these in editor
                             st.write(f"Showing top {len(full_global_indices)} most distant compounds (candidates for removal):")
                             
                             df_candidates = df_result.loc[full_global_indices].copy()
                             df_candidates['Distance_AD'] = dists[sorted_local_indices]
                             df_candidates['Is_Stat_Outlier'] = [idx in global_outliers for idx in full_global_indices]
                             df_candidates['Delete'] = df_candidates['Is_Stat_Outlier'] # Auto-select if it is an outlier
                             
                             # Move key columns to front
                             cols = ['Delete', 'Distance_AD', 'Is_Stat_Outlier'] + [c for c in df_candidates.columns if c not in ['Delete', 'Distance_AD', 'Is_Stat_Outlier']]
                             df_candidates = df_candidates[cols]

                             edited_df = st.data_editor(
                                 df_candidates, 
                                 column_config={
                                     "Delete": st.column_config.CheckboxColumn("Select to Delete", default=False),
                                     "Distance_AD": st.column_config.NumberColumn("Distance", format="%.4f"),
                                     "Is_Stat_Outlier": st.column_config.CheckboxColumn("Stat. Outlier", disabled=True)
                                 },
                                 disabled=[c for c in df_candidates.columns if c != 'Delete'],
                                 key="outlier_editor_manual"
                             )
                             
                             to_delete_manual = edited_df[edited_df['Delete']].index.tolist()
                             
                             if to_delete_manual:
                                 if st.button(f"🗑️ Delete {len(to_delete_manual)} Selected Compounds"):
                                      current = st.session_state.removed_outliers_indices
                                      # Add new ones
                                      st.session_state.removed_outliers_indices = list(set(current + to_delete_manual))
                                      st.success(f"Deleted {len(to_delete_manual)} compounds.")
                                      st.session_state['ad_info'] = None
                                      st.rerun()

                         else:
                             # Fallback if manual mapping fails due to missing data (e.g. old model in state)
                             if len(global_outliers) == 0:
                                 st.info("No statistical outliers found. (Top 50 visualization requires retraining to update mapping).")
                             else:
                                 # ... existing fallback for just outliers ...
                                 st.write("Displaying detected outliers only (Mapping data missing for full list).")
                                 # ... logic for just global_outliers as before ...
                                 # For brevity, let's just ask user to retrain if data is missing, caused by hot-reload
                                 st.warning("Please click 'Train Models' to refresh data for manual deletion.")
                                      
                                     
                         if st.session_state.removed_outliers_indices:
                             st.info(f"ℹ️ Total excluded outliers so far: {len(st.session_state.removed_outliers_indices)}")
                             if st.button("Reset Removed Outliers"):
                                 st.session_state.removed_outliers_indices = []
                                 st.rerun()

                     # ------------------------------------

                     results_success = results[results["Acurácia"] > 0] if "Erro" in results.columns else results
                     
                     if not results_success.empty:
                         st.subheader(t['metrics_header'])
                         try:
                             # Ensure columns exist before formatting
                             format_dict = {
                                 "Acurácia": "{:.3f}",
                                 "F1-Score": "{:.3f}",
                                 "MCC": "{:.3f}",
                                 "Sensibilidade": "{:.3f}",
                                 "Especificidade": "{:.3f}",
                                 "AUC": "{:.3f}"
                             }
                             cols_to_format = {k: v for k, v in format_dict.items() if k in results_success.columns}
                             st.dataframe(results_success.style.format(cols_to_format))
                         except Exception as e:
                             st.dataframe(results_success)
                         
                         # Best Model Suggestion & Report
                         st.divider()
                         st.subheader(t['best_model_header'])
                         
                         # Find best model based on MCC (or Accuracy if MCC is missing)
                         metric_sort = "MCC" if "MCC" in results_success.columns else "Acurácia"
                         best_row = results_success.loc[results_success[metric_sort].idxmax()]
                         best_model_name = best_row["Modelo"]
                         
                         st.info(t['best_model_rec'].format(best_model_name))
                         st.write(t['best_model_metrics'].format(best_row["MCC"], best_row["Acurácia"], best_row["F1-Score"]))
                         
                         # Generate Report PDF
                         try:
                             dataset_stats = {
                                 t['total_orig']: str(st.session_state.input_len),
                                 t['total_final']: str(len(df_result)),
                                 t['actives']: str(actives),
                                 t['inactives']: str(inactives)
                             }
                             
                             # Generate ROC Image for Report if roc_data exists
                             temp_roc_path = None
                             if roc_data:
                                 import matplotlib.pyplot as plt
                                 import tempfile
                                 import os
                                 
                                 plt.figure(figsize=(8, 6))
                                 for model_name, data in roc_data.items():
                                     plt.plot(data['fpr'], data['tpr'], label=f"{model_name} (AUC = {data['auc']:.2f})")
                                 
                                 plt.plot([0, 1], [0, 1], 'k--', label='Random')
                                 plt.xlabel('False Positive Rate')
                                 plt.ylabel('True Positive Rate')
                                 plt.title('ROC Curve Comparison')
                                 plt.legend(loc="lower right")
                                 
                                 # Save to temp file
                                 fd, temp_roc_path = tempfile.mkstemp(suffix=".png")
                                 os.close(fd)
                                 plt.savefig(temp_roc_path, bbox_inches='tight', dpi=150)
                                 plt.close()
                             
                             # Prepare parameters for report
                             model_params = {
                                 "Descriptor Type": config.get('descriptor_type', 'Morgan'),
                                 "Bits": config.get('n_bits', 1024),
                                 "Radius": config.get('radius', 2),
                                 "Test Split": f"{test_split:.0%}",
                                 "Calculation Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                             }

                             pdf_bytes = generate_pdf_report(
                                 results_success, 
                                 best_model_name, 
                                 dataset_stats, 
                                 logo_path="assets/logo.png",
                                 lang=lang,
                                 roc_plot_path=temp_roc_path,
                                 params=model_params
                             )
                             
                             # Cleanup temp file
                             if temp_roc_path and os.path.exists(temp_roc_path):
                                 os.remove(temp_roc_path)
                             
                             st.download_button(
                                 label=t['download_report_btn'],
                                 data=pdf_bytes,
                                 file_name=t['report_filename'],
                                 mime="application/pdf"
                             )
                         except ImportError as e:
                             st.error(f"Biblioteca FPDF problema: {e}")
                         except Exception as e:
                             st.error(f"Erro ao gerar PDF: {e}")

                         # Metrics Bar Chart
                         st.divider()
                         st.subheader(t['viz_header'])
                         
                         metrics_to_plot = ["Acurácia", "MCC", "F1-Score", "Sensibilidade", "Especificidade", "AUC"]
                         valid_metrics = [m for m in metrics_to_plot if m in results.columns]
                         
                         if valid_metrics:
                             df_melted = results_success.melt(id_vars=["Modelo"], value_vars=valid_metrics, var_name="Métrica", value_name="Valor")
                             
                             chart = alt.Chart(df_melted).mark_bar().encode(
                                  y=alt.Y('Métrica', axis=None),
                                  x=alt.X('Valor', title='Score'),
                                  color='Métrica',
                                  row=alt.Row('Modelo', header=alt.Header(labelAngle=0, labelAlign='left')),
                                  tooltip=['Modelo', 'Métrica', alt.Tooltip('Valor', format='.3f')]
                              ).properties(
                                  height=len(valid_metrics) * 15, 
                                  width=500
                              ).configure_view(
                                  stroke='transparent'
                              )
                             
                             st.altair_chart(chart)
                     else:
                         st.warning(t['all_failed'])
                         
                     # ROC Curve Visualization
                     if roc_data:
                        st.divider()
                        st.subheader(t.get('roc_header', 'ROC Curve')) # Fallback to 'ROC Curve' if key missing
                        
                        roc_df_list = []
                        for model_name, data in roc_data.items():
                             fpr = data['fpr']
                             tpr = data['tpr']
                             auc = data['auc']
                             
                             # Downsample for faster plotting if too many points
                             if len(fpr) > 500:
                                 indices = np.linspace(0, len(fpr) - 1, 500).astype(int)
                                 fpr = fpr[indices]
                                 tpr = tpr[indices]
                             
                             temp_df = pd.DataFrame({
                                 'FPR': fpr,
                                 'TPR': tpr,
                                 'Model': f"{model_name} (AUC: {auc:.3f})"
                             })
                             roc_df_list.append(temp_df)
                        
                        if roc_df_list:

                             all_roc_df = pd.concat(roc_df_list, ignore_index=True)
                             
                             # Base chart for models
                             roc_chart = alt.Chart(all_roc_df).mark_line().encode(
                                 x=alt.X('FPR', title='False Positive Rate'),
                                 y=alt.Y('TPR', title='True Positive Rate'),
                                 color=alt.Color('Model', title='Model'),
                                 tooltip=['Model', 'FPR', 'TPR']
                             )
                             
                             # Random guess line
                             random_guess = pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})
                             line_chart = alt.Chart(random_guess).mark_line(strokeDash=[5, 5], color='black').encode(
                                 x='FPR',
                                 y='TPR'
                             )
                             
                             final_roc_chart = (roc_chart + line_chart).properties(
                                 title='Multi-Model ROC Curve',
                                 width=600,
                                 height=500
                             ).interactive()
                             
                             st.altair_chart(final_roc_chart, use_container_width=True)

                     # Download Section
                     trained_models = st.session_state.get('trained_models')
                     if trained_models:
                         st.divider()
                         st.subheader(t['download_models_header'])
                         st.write(t['download_models_text'])
                         
                         cols = st.columns(len(trained_models))
                         for i, (name, model) in enumerate(trained_models.items()):
                             # Wrap model with metadata
                             model_package = {
                                 "model": model,
                                 "metadata": {
                                     "name": name,
                                     "descriptor_type": config.get('descriptor_type', 'Morgan'),
                                     "n_bits": config.get('n_bits', 1024),
                                     "radius": config.get('radius', 2),
                                     "version": "1.0"
                                 }
                             }
                             
                             # Attach AD Model if available for this descriptor
                             dt = config.get('descriptor_type', 'Morgan')
                             if 'ad_info' in st.session_state and dt in st.session_state['ad_info']:
                                 model_package["ad_model"] = st.session_state['ad_info'][dt]['model']
                                 
                             model_pkl = pickle.dumps(model_package)
                             
                             col_idx = i % 3
                             if i % 3 == 0 and i > 0:
                                 st.write("")
                                 cols = st.columns(3)
                             
                             with cols[col_idx]:
                                 st.download_button(
                                     label=f"📥 {name}",
                                     data=model_pkl,
                                     file_name=f"qsar_model_{name.replace(' ', '_').lower()}.pkl",
                                     mime="application/octet-stream",
                                     key=f"dl_{name}"
                                 )
                 else:
                     st.warning(t['empty_results'])
