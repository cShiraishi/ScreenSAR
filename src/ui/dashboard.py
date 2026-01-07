import streamlit as st
import pandas as pd
import altair as alt
import io
import pickle
from src.core.curation import CuradoriaQSAR
from src.core.modeling import ModeladorQSAR
from src.utils.report import create_pdf_report

def render_dashboard(config):
    """
    Renders the main dashboard results (Curated Data, Charts, Modeling).
    """
    t = config['t']
    lang = config['lang']
    
    if st.session_state.curated_result is not None:
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
            n_bits = col_fp1.selectbox(t['nbits'], [1024, 2048, 512], index=0)
            radius = col_fp2.number_input(t['radius'], min_value=1, max_value=4, value=2)
            
            if st.button(t['gen_map_btn']):
                try:
                    from sklearn.decomposition import PCA
                    
                    with st.spinner("Generating..."):
                        curador_pca = CuradoriaQSAR(df_result)
                        fps, valid_idxs = curador_pca.gerar_fingerprints(df_result, n_bits=n_bits, radius=radius)
                        
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

        st.markdown("---")
        st.subheader(t['model_header'])
        
        with st.expander(t['model_expander'], expanded=False):
            st.write(t['model_intro'])
            
            # MODI Check
            if st.button(t['calc_modi']):
                with st.spinner(t['modi_spinner']):
                    modelador_modi = ModeladorQSAR(df_result)
                    X_modi, y_modi = modelador_modi.gerar_dados()
                    
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
            
            train_btn = st.button(t['train_btn'])
            
            if train_btn:
                if not selected_models:
                    st.warning(t['warn_select_model'])
                else:
                    try:
                        with st.spinner(t['training_spinner']):
                            modelador = ModeladorQSAR(df_result)
                            X, y = modelador.gerar_dados(n_bits=1024, radius=2)
                            
                            if len(y) < 20:
                                 st.error(t['error_insufficient'])
                            else:
                                results, trained_models, roc_data = modelador.treinar_avaliar(
                                    X, y, selected_models, test_size=test_split
                                )
                                
                                st.session_state['modeling_results'] = results
                                st.session_state['roc_data'] = roc_data
                                st.session_state['trained_models'] = trained_models
                                
                                st.success(t['training_success'])

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
                             
                             pdf_bytes = create_pdf_report(
                                 results_success, 
                                 best_model_name, 
                                 dataset_stats, 
                                 logo_path="logo.png",
                                 lang=lang
                             )
                             
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
                                 x=alt.X('Métrica', axis=None),
                                 y=alt.Y('Valor', title='Score'),
                                 color='Métrica',
                                 column=alt.Column('Modelo', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                                 tooltip=['Modelo', 'Métrica', alt.Tooltip('Valor', format='.3f')]
                             ).properties(
                                 width=80
                             ).configure_view(
                                 stroke='transparent'
                             )
                             
                             st.altair_chart(chart)
                     else:
                         st.warning(t['all_failed'])
                         
                     # Download Section
                     trained_models = st.session_state.get('trained_models')
                     if trained_models:
                         st.divider()
                         st.subheader(t['download_models_header'])
                         st.write(t['download_models_text'])
                         
                         cols = st.columns(len(trained_models))
                         for i, (name, model) in enumerate(trained_models.items()):
                             model_pkl = pickle.dumps(model)
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
