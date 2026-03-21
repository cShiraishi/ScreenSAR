import streamlit as st
import time
import pandas as pd
import pickle
import numpy as np
from contextlib import contextmanager
@contextmanager
def patched_logistic_regression():
    """Context manager to handle backward compatibility for LogisticRegression models."""
    from sklearn.linear_model import LogisticRegression
    original_setstate = getattr(LogisticRegression, '__setstate__', None)

    def new_setstate(self, state):
        if 'multi_class' not in state:
            state['multi_class'] = 'ovr' # Default for older versions (<=0.19)
        if original_setstate:
            original_setstate(self, state)
        else:
            self.__dict__.update(state)

    LogisticRegression.__setstate__ = new_setstate
    try:
        yield
    finally:
        if original_setstate:
            LogisticRegression.__setstate__ = original_setstate
        else:
            del LogisticRegression.__setstate__



def render_prediction_page(config):
    t = config['t']
    
    # Initialize Session State
    if 'pred_result_df' not in st.session_state:
        st.session_state.pred_result_df = None
    if 'pred_stats' not in st.session_state:
        st.session_state.pred_stats = {}
    
    # Logo Emphasis
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("assets/logo.png", use_container_width=True)
        
    st.title(t['pred_title'])
    
    st.markdown(t['pred_intro'])
    
    # 1. Upload Model
    st.subheader(t['pred_step1'])
    uploaded_model = st.file_uploader(t['upload_model_label'], type=['pkl'], key='model_loader')
    
    model_data = None
    if uploaded_model:
        try:
            with patched_logistic_regression():
                model_package = pickle.load(uploaded_model)

            
            # Check if it's the new format (dict with metadata) or old (raw model)
            if isinstance(model_package, dict) and "metadata" in model_package:
                model = model_package['model']
                metadata = model_package['metadata']
                ad_model = model_package.get('ad_model') # New field
                
                st.success(t['model_loaded'].format(metadata.get('name', 'Unknown')))
                st.info(t['model_config'].format(metadata.get('descriptor_type', 'Morgan'), metadata.get('n_bits'), metadata.get('radius')))
                if ad_model:
                     st.info(f"🛡️ Applicability Domain Model Loaded (Threshold: {ad_model.threshold_AD:.3f})")
                     
                model_data = {"model": model, "meta": metadata, "ad_model": ad_model}
            else:
                # Legacy fallback
                st.warning(t['legacy_warn'])
                model_data = {
                    "model": model_package,
                    "meta": {"descriptor_type": "Morgan", "n_bits": 1024, "radius": 2}
                }
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
            
    # 2. Upload Molecules
    st.divider()
    st.subheader(t['pred_step2'])
    tab_upload, tab_paste = st.tabs(["📂 " + t.get('tab_upload', "Upload File"), "📝 " + t.get('tab_paste', "Paste SMILES")])
    
    df_mols = None
    smiles_col = "SMILES" # Default name for pasted data
    
    # SOURCE 1: FILE UPLOAD
    with tab_upload:
        uploaded_mols_list = st.file_uploader(t['upload_mols_label'], type=['csv', 'txt', 'xlsx'], accept_multiple_files=True)
        
        if uploaded_mols_list:
            all_dfs = []
            files_to_load = uploaded_mols_list if isinstance(uploaded_mols_list, list) else [uploaded_mols_list]
            
            for uploaded_mols in files_to_load:
                try:
                    if uploaded_mols.name.endswith('.csv') or uploaded_mols.name.endswith('.txt'):
                         try:
                            import io
                            import csv
                            buf = io.BytesIO(uploaded_mols.getvalue())
                            # Tentar descobrir o separador rapidamente
                            sample = buf.read(10240).decode('utf-8', errors='ignore')
                            buf.seek(0)
                            try:
                                sep = csv.Sniffer().sniff(sample).delimiter
                            except:
                                sep = ','
                            df_temp = pd.read_csv(buf, sep=sep, engine='c', on_bad_lines='skip')
                         except Exception:
                            try:
                                buf = io.BytesIO(uploaded_mols.getvalue())
                                df_temp = pd.read_csv(buf, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                            except Exception:
                                pass
                    else:
                         df_temp = pd.read_excel(uploaded_mols)
                    
                    if not df_temp.empty:
                        all_dfs.append(df_temp)
                        
                except Exception as e:
                    st.error(f"Error reading file {uploaded_mols.name}: {e}")
            
            if all_dfs:
                try:
                    df_mols = pd.concat(all_dfs, ignore_index=True)
                    
                    # Try to find SMILES column for FILE
                    found_col = None
                    for c in df_mols.columns:
                        if "SMILES" in c.upper() or "SMILE" in c.upper() or "STRUCTURE" in c.upper():
                            found_col = c
                            break
                    
                    if not found_col and len(df_mols.columns) == 1:
                        found_col = df_mols.columns[0] # Assume single column list
                    
                    smiles_col = found_col
                    
                    st.success(f"Successfully loaded {len(all_dfs)} file(s) with {len(df_mols)} total molecules.")
                except Exception as e:
                    st.error(f"Error merging files: {e}")

    # SOURCE 2: PASTE SMILES
    with tab_paste:
        st.write(t.get('paste_instruction', "Paste SMILES codes, one per line."))
        pasted_text = st.text_area("SMILES Input", height=200, placeholder="C1=CC=CC=C1\nCCCCC\n...")
        
        if pasted_text:
            lines = [l.strip() for l in pasted_text.split('\n') if l.strip()]
            if lines:
                df_mols = pd.DataFrame({'SMILES': lines})
                smiles_col = 'SMILES'
    
    # COMMON PROCESSING
    if df_mols is not None:
        if smiles_col:
                st.write(t['analyzed_mols'].format(len(df_mols) if not isinstance(df_mols, pd.io.parsers.TextFileReader) else "Large Dataset (Chunked)", smiles_col))
                
                if st.button(t['run_pred_btn']):
                    if not model_data:
                        st.error("Please upload a model first (Step 1).")
                    else:
                        # Prepare data
                        meta = model_data['meta']
                        training_smiles_set = set(meta.get('training_smiles', []))
                        
                        from rdkit import Chem
                        from rdkit.Chem import AllChem, MACCSkeys
                        
                        # Configuration for chunks
                        CHUNK_SIZE = 5000 
                        
                        # Helper generator to yield chunks
                        def get_chunks(source, chunk_size):
                            if isinstance(source, pd.io.parsers.TextFileReader):
                                for chunk in source:
                                    yield chunk
                            elif isinstance(source, pd.DataFrame):
                                total_rows = len(source)
                                for i in range(0, total_rows, chunk_size):
                                    yield source.iloc[i:i+chunk_size]
                            else:
                                raise ValueError("Unsupported data source")

                        # Use the already loaded dataframe (df_mols) as source
                        # This avoids re-reading the file stream which can cause I/O closed errors,
                        # and uses the encoding logic that already succeeded during preview.
                        data_source = df_mols
                        total_steps = len(df_mols)
                        
                        # Progress Bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = time.time()
                        
                        all_results = []
                        total_processed = 0
                        total_active = 0
                        
                        # Iterate
                        chunk_idx = 0
                        total_scanned = 0
                        
                        for df_chunk in get_chunks(data_source, CHUNK_SIZE):
                            chunk_idx += 1
                            current_chunk_len = len(df_chunk)
                            total_scanned += current_chunk_len
                            
                            # Update progress with percentage
                            if total_steps > 0:
                                progress_val = min(total_scanned / total_steps, 1.0)
                                progress_bar.progress(progress_val)
                            
                            elapsed = time.time() - start_time
                            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                            status_text.text(f"Processing chunk {chunk_idx}... (Scanned: {total_scanned} | Time: {elapsed_str})")
                            
                            # Valid rows
                            valid_rows = []
                            fps = []
                            in_training = []
                            
                            for idx, row in df_chunk.iterrows():
                                smi = row.get(smiles_col)
                                if pd.isna(smi): continue
                                
                                try:
                                    mol = Chem.MolFromSmiles(str(smi))
                                    if mol:
                                        desc_type = meta.get('descriptor_type', 'Morgan')
                                        n_bits = meta.get('n_bits', 1024)
                                        radius = meta.get('radius', 2)
                                        
                                        if desc_type == "MACCS":
                                            fp = MACCSkeys.GenMACCSKeys(mol)
                                        elif desc_type == "RDKit":
                                            fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=n_bits, nBitsPerHash=2)
                                        else:
                                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                                            
                                        fps.append(np.array(fp))
                                        valid_rows.append(row)
                                        
                                        if training_smiles_set:
                                            can_smi = Chem.MolToSmiles(mol)
                                            in_training.append("Yes" if can_smi in training_smiles_set else "No")
                                        else:
                                            in_training.append("Unknown")
                                except:
                                    pass
                            
                            if fps:
                                X_pred = np.array(fps)
                                X_pred = np.array(fps)
                                y_pred = model_data['model'].predict(X_pred)
                                y_proba = model_data['model'].predict_proba(X_pred)[:, 1] if hasattr(model_data['model'], 'predict_proba') else [0]*len(y_pred)
                                
                                # AD Check
                                ad_inside = [None] * len(y_pred)
                                ad_dist = [None] * len(y_pred)
                                
                                if model_data.get('ad_model'):
                                    try:
                                        is_inside, distances = model_data['ad_model'].predict(X_pred)
                                        ad_inside = ["Inside" if x else "Outside" for x in is_inside]
                                        ad_dist = distances
                                    except Exception as e:
                                        # Handle cases where AD model might fail (e.g. dimension mismatch)
                                        # st.warning(f"AD Check failed: {e}")
                                        pass
                                
                                # Create Chunk Result
                                df_res_chunk = pd.DataFrame(valid_rows)
                                df_res_chunk['Predicted_Class'] = y_pred
                                df_res_chunk['Probability_Active'] = y_proba
                                df_res_chunk['Prediction_Label'] = ["Active" if x==1 else "Inactive" for x in y_pred]
                                df_res_chunk['In_Training_Set'] = in_training
                                
                                if model_data.get('ad_model'):
                                    df_res_chunk['AD_Status'] = ad_inside
                                    df_res_chunk['AD_Distance'] = ad_dist
                                
                                all_results.append(df_res_chunk)
                                total_processed += len(df_res_chunk)
                                total_active += sum(y_pred)
                            
                            # Clean up memory
                            del fps
                            del valid_rows
                            
                        progress_bar.progress(1.0)
                        total_elapsed = time.time() - start_time
                        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
                        status_text.success(f"Processing complete in {elapsed_str}!")
                        
                        if all_results:
                            final_df = pd.concat(all_results, ignore_index=True)
                            # Store in session state
                            st.session_state.pred_result_df = final_df
                            st.session_state.pred_stats = {'total_active': total_active, 'total_processed': total_processed, 'time_elapsed': elapsed_str}
                        else:
                            st.error(t['error_no_descriptors'])
                            st.session_state.pred_result_df = None

        else:
            st.error(t['error_smiles_col'])
            
    # RENDER RESULTS FROM SESSION STATE (Outside the button click)
    if st.session_state.pred_result_df is not None:
        final_df = st.session_state.pred_result_df
        stats = st.session_state.pred_stats
        
        st.divider()
        st.subheader(t['pred_results_title'])
        st.write(t['pred_summary'].format(stats.get('total_active', 0), stats.get('total_processed', 0)))
        
        if 'time_elapsed' in stats:
             st.info(f"⏱️ Screening completed in: **{stats['time_elapsed']}**")
        
        # New AD Metrics
        if 'AD_Status' in final_df.columns:
            total_inside = final_df['AD_Status'].value_counts().get('Inside', 0)
            total_outside = final_df['AD_Status'].value_counts().get('Outside', 0)
            
            c_ad1, c_ad2, c_ad3 = st.columns(3)
            c_ad1.metric("🛡️ Inside Domain", f"{total_inside}", help="Reliable predictions")
            c_ad2.metric("⚠️ Outside Domain", f"{total_outside}", help="Unreliable predictions")
            c_ad3.metric("Coverage", f"{total_inside/len(final_df):.1%}" if len(final_df)>0 else "0%")
            
            if total_outside > 0:
                st.caption(f"Note: {total_outside} compounds are structurally distinct from the training set. Consider filtering them out.")
        
        with st.expander("ℹ️ Help: Understanding Confidence & Applicability Domain", expanded=True):
            st.markdown("""
            *   **Confidence (Probability_Active)**: The model's estimated probability that the compound is **Active**. 
                *   Values closer to **1.0** indicate high confidence in Activity.
                *   Values closer to **0.0** indicate high confidence in Inactivity.
            *   **AD Status**: Indicates if the compound is within the **Applicability Domain**.
                *   ✅ **Inside**: The compound is chemically similar to the training set. The prediction is reliable.
                *   ⚠️ **Outside**: The compound is structurally distinct from the training data. The prediction is less reliable.
            *   **AD Distance**: The computed distance to the nearest training neighbors. Lower values mean higher similarity.
            *   **In_Training_Set**: Indicates if the exact same molecule was used to *train* the model. If 'Yes', a high prediction confidence might simply be the model remembering the training data, so it is not a *novel* discovery.
            """)
        
        # COLUMN CONFIGURATION & ORDERING
        display_cols = [c for c in final_df.columns if c not in ['SMILES', 'Molecule ChEMBL ID']]
        # Prioritize important cols
        priority = ['SMILES', 'Prediction_Label', 'Probability_Active', 'In_Training_Set', 'AD_Status', 'AD_Distance']
        
        # Reorder: Priority first, then others
        ordered_cols = [c for c in priority if c in final_df.columns]
        remaining = [c for c in final_df.columns if c not in ordered_cols]
        final_view = final_df[ordered_cols + remaining]
        
        if len(final_view) > 5000:
            st.info(f"⚠️ Displaying only the first 5000 out of {len(final_view)} records to prevent your browser from crashing/freezing. You can download the **full** dataset below.")
            df_to_display = final_view.head(5000)
        else:
            df_to_display = final_view

        st.dataframe(
            df_to_display,
            column_config={
                "AD_Status": st.column_config.TextColumn(
                    "AD Status",
                    help="Applicability Domain Status",
                    validate="^(Inside|Outside)$"
                ),
                "Probability_Active": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                    help="Probability of the compound being Active (0.0 to 1.0). Higher values indicate greater certainty by the model."
                ),
                "AD_Distance": st.column_config.NumberColumn(
                    "AD Distance",
                    format="%.3f"
                )
            },
            use_container_width=True
        )
        
        st.divider()
        st.subheader("🔍 " + (t.get('filter_header', 'Filter by Confidence')))
        
        # 1. Probability Slider
        threshold = st.slider(
            t.get('prob_threshold', 'Probability Threshold (Active class)'), 
            min_value=0.5, 
            max_value=0.99, 
            value=0.7, 
            step=0.05,
            help="Filter molecules with Probability_Active >= Threshold"
        )
        
        # 1.1 AD Filter
        filter_ad = False
        if 'AD_Status' in final_df.columns:
            filter_ad = st.checkbox("🛡️ Only show molecules INSIDE Applicability Domain", value=False)
        
        # 2. Filter
        df_filtered = final_df[final_df['Probability_Active'] >= threshold]
        
        if filter_ad:
            df_filtered = df_filtered[df_filtered['AD_Status'] == "Inside"]
        
        st.write(f"**Molecules selected:** {len(df_filtered)} / {len(final_df)}")
        
        if not df_filtered.empty:
            # Reorder for filtered view too
            ordered_cols = [c for c in priority if c in df_filtered.columns]
            remaining = [c for c in df_filtered.columns if c not in ordered_cols]
            filtered_view = df_filtered[ordered_cols + remaining]
            
            st.dataframe(
                filtered_view.head(50), 
                column_config={
                    "AD_Status": st.column_config.TextColumn(
                        "AD Status",
                        help="Applicability Domain Status",
                        validate="^(Inside|Outside)$"
                    ),
                    "Probability_Active": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                        help="Probability of the compound being Active (0.0 to 1.0). Higher values indicate greater certainty by the model."
                    )
                },
                use_container_width=True
            )
        else:
            st.warning("No molecules found with this threshold.")
        
        # Download
        # Warning for massive files
        if len(final_df) > 100000:
            st.warning("Large result set. Converting to CSV might take a moment.")
            
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        col_dl1.download_button(t['download_pred'], csv, "prediction_results_full.csv", "text/csv")
        
        # Download Active Only
        df_active = final_df[final_df['Predicted_Class'] == 1]
        if not df_active.empty:
            csv_active = df_active.to_csv(index=False).encode('utf-8')
            col_dl2.download_button("Download All Actives", csv_active, "prediction_results_actives_only.csv", "text/csv")

        # Download Filtered High Conf
        if not df_filtered.empty:
            csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
            col_dl3.download_button(
                f"📥 Download High Confidence (>{threshold})", 
                csv_filtered, 
                f"prediction_high_conf_{threshold}.csv", 
                "text/csv",
                type="primary"
            )

        # PDF Report
        st.divider()
        st.subheader(t.get('gen_pdf_header', "📄 Generate PDF Report"))
        st.write(t.get('gen_pdf_desc', "Generate a summary PDF report highlighting the top Active hits that fall **Inside the Applicability Domain** and their principal scaffolds."))
        
        try:
            from src.utils.report import generate_prediction_report
            model_name_for_report = model_data['meta'].get('name', 'QSAR Model') if model_data else 'QSAR Model'
            
            pdf_bytes = generate_prediction_report(
                final_df, 
                stats, 
                model_name=model_name_for_report,
                logo_path="assets/logo.png",
                lang=config.get('lang', 'English')
            )
            
            if pdf_bytes:
                st.download_button(
                    label="📄 Download Prediction Report (PDF)",
                    data=pdf_bytes,
                    file_name="virtual_screening_report.pdf",
                    mime="application/pdf",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error generating PDF report: {e}")
