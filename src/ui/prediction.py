import streamlit as st
import time
import pandas as pd
import pickle
import numpy as np
from src.core.curation import CuradoriaQSAR
from contextlib import contextmanager
from sklearn.linear_model import LogisticRegression

@contextmanager
def patched_logistic_regression():
    """Context manager to handle backward compatibility for LogisticRegression models."""
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
                st.success(t['model_loaded'].format(metadata.get('name', 'Unknown')))
                st.info(t['model_config'].format(metadata.get('descriptor_type', 'Morgan'), metadata.get('n_bits'), metadata.get('radius')))
                model_data = {"model": model, "meta": metadata}
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
    if model_data:
        st.divider()
        st.subheader(t['pred_step2'])
        uploaded_mols = st.file_uploader(t['upload_mols_label'], type=['csv', 'txt', 'xlsx'])
        
        if uploaded_mols:
            try:
                if uploaded_mols.name.endswith('.csv') or uploaded_mols.name.endswith('.txt'):
                     try:
                        df_mols = pd.read_csv(uploaded_mols, sep=None, engine='python')
                     except UnicodeDecodeError:
                        uploaded_mols.seek(0)
                        df_mols = pd.read_csv(uploaded_mols, sep=None, engine='python', encoding='latin1')
                else:
                     df_mols = pd.read_excel(uploaded_mols)
                
                # Try to find SMILES column
                cols = [c.upper() for c in df_mols.columns]
                smiles_col = None
                for c in df_mols.columns:
                    if "SMILES" in c.upper() or "SMILE" in c.upper() or "STRUCTURE" in c.upper():
                        smiles_col = c
                        break
                
                if not smiles_col and len(df_mols.columns) == 1:
                    smiles_col = df_mols.columns[0] # Assume single column list
                
                if smiles_col:
                    st.write(t['analyzed_mols'].format(len(df_mols) if not isinstance(df_mols, pd.io.parsers.TextFileReader) else "Large Dataset (Chunked)", smiles_col))
                    
                    if st.button(t['run_pred_btn']):
                        # Prepare data
                        meta = model_data['meta']
                        
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

                        # Re-initialize reader for CSV if it was already consumed or needed to be cleanly read
                        uploaded_mols.seek(0)
                        
                        is_csv = uploaded_mols.name.endswith('.csv') or uploaded_mols.name.endswith('.txt')
                        
                        if is_csv:
                             # Count lines to estimate total
                             try:
                                total_steps = sum(1 for line in uploaded_mols) - 1 # Subtract header
                             except:
                                total_steps = 0
                             uploaded_mols.seek(0)
                             
                             # Read as iterator
                             try:
                                 data_source = pd.read_csv(uploaded_mols, sep=None, engine='python', chunksize=CHUNK_SIZE)
                                 # Trigger a read to catch potential immediate encoding errors
                                 # Actually read_csv with chunksize returns an iterator, it might not fail until iteration
                                 # But if it fails, we catch it in the loop or we can try to peek?
                                 # A safer way allows retrying the whole block.
                             except:
                                 pass # Fallthrough to retry logic below isn't easy with this structure, let's just make the assignment robust

                             # Let's use a robust approach
                             uploaded_mols.seek(0)
                             try:
                                 # Try UTF-8 first (default)
                                 data_source = pd.read_csv(uploaded_mols, sep=None, engine='python', chunksize=CHUNK_SIZE)
                                 # Test first chunk
                                 next(data_source) 
                                 uploaded_mols.seek(0)
                                 data_source = pd.read_csv(uploaded_mols, sep=None, engine='python', chunksize=CHUNK_SIZE)
                             except UnicodeDecodeError:
                                 uploaded_mols.seek(0)
                                 data_source = pd.read_csv(uploaded_mols, sep=None, engine='python', chunksize=CHUNK_SIZE, encoding='latin1')
                             except Exception:
                                 # Fallback to latin1 just in case other errors masked it, or re-raise
                                 uploaded_mols.seek(0)
                                 data_source = pd.read_csv(uploaded_mols, sep=None, engine='python', chunksize=CHUNK_SIZE, encoding='latin1')
                        else:
                             # Excel is already read into memory as df_mols because we can't chunk read easily
                             # So we just chunk the dataframe
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
                                except:
                                    pass
                            
                            if fps:
                                X_pred = np.array(fps)
                                y_pred = model_data['model'].predict(X_pred)
                                y_proba = model_data['model'].predict_proba(X_pred)[:, 1] if hasattr(model_data['model'], 'predict_proba') else [0]*len(y_pred)
                                
                                # Create Chunk Result
                                df_res_chunk = pd.DataFrame(valid_rows)
                                df_res_chunk['Predicted_Class'] = y_pred
                                df_res_chunk['Probability_Active'] = y_proba
                                df_res_chunk['Prediction_Label'] = ["Active" if x==1 else "Inactive" for x in y_pred]
                                
                                all_results.append(df_res_chunk)
                                total_processed += len(df_res_chunk)
                                total_active += sum(y_pred)
                            
                            # Clean up memory
                            del fps
                            del valid_rows
                            
                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")
                        
                        if all_results:
                            final_df = pd.concat(all_results, ignore_index=True)
                            # Store in session state
                            st.session_state.pred_result_df = final_df
                            st.session_state.pred_stats = {'total_active': total_active, 'total_processed': total_processed}
                        else:
                            st.error(t['error_no_descriptors'])
                            st.session_state.pred_result_df = None
                else:
                    st.error(t['error_smiles_col'])
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
        # RENDER RESULTS FROM SESSION STATE (Outside the button click)
        if st.session_state.pred_result_df is not None:
            final_df = st.session_state.pred_result_df
            stats = st.session_state.pred_stats
            
            st.divider()
            st.subheader(t['pred_results_title'])
            st.write(t['pred_summary'].format(stats.get('total_active', 0), stats.get('total_processed', 0)))
            
            st.dataframe(final_df.head())
            
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
            
            # 2. Filter
            df_filtered = final_df[final_df['Probability_Active'] >= threshold]
            
            st.write(f"**Molecules selected:** {len(df_filtered)} / {len(final_df)}")
            
            if not df_filtered.empty:
                st.dataframe(df_filtered.head())
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
