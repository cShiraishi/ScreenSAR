import streamlit as st
import pandas as pd
from src.core.curation import CuradoriaQSAR
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard

def read_single_file(uploaded_file):
    """
    Helper function to read a single uploaded file (CSV or Excel)
    and return a DataFrame.
    """
    # Check for Excel magic bytes even if named .csv
    # XLSX starts with PK (50 4B), XLS starts with D0 CF
    file_start = uploaded_file.read(4)
    uploaded_file.seek(0)
    
    is_likely_excel = False
    if file_start.startswith(b'PK') or file_start.startswith(b'\xd0\xcf'):
        is_likely_excel = True
    
    df_input = None
    
    if uploaded_file.name.endswith('.csv') and not is_likely_excel:
        try:
            # First attempt: Auto-detect separator with python engine, utf-8
            df_input = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            uploaded_file.seek(0)
            try:
                # Second attempt: Auto-detect separator with python engine, latin1
                df_input = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1')
            except Exception:
                uploaded_file.seek(0)
                # Fallback: specific separators if auto-detection fails
                try:
                    df_input = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
                except:
                    uploaded_file.seek(0)
                    try:
                        df_input = pd.read_csv(uploaded_file, sep=',', encoding='latin1')
                    except:
                        # Final Resort: Skip bad lines
                        uploaded_file.seek(0)
                        df_input = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                        st.warning(f"⚠️ Warning: Potentially malformed lines were skipped in {uploaded_file.name}.")
    else:
        # Read as Excel if extension says so OR if magic bytes matched
        try:
            # Explicitly determine engine based on signature if possible
            engine_arg = None
            if file_start.startswith(b'PK'):
                engine_arg = 'openpyxl'
            elif file_start.startswith(b'\xd0\xcf'):
                 # Old .xls files usually need 'xlrd' or default
                engine_arg = 'xlrd'
            
            if engine_arg:
                df_input = pd.read_excel(uploaded_file, engine=engine_arg)
            else:
                df_input = pd.read_excel(uploaded_file)

        except Exception as e_xls:
             # If it failed as Excel but was .csv, maybe it really was a weird CSV? 
             # But if is_likely_excel was True, we probably failed on actual Excel logic.
             if is_likely_excel:
                 # It might be a ZIPPED CSV file (which starts with PK) but is not an Excel file.
                 try:
                     uploaded_file.seek(0)
                     df_input = pd.read_csv(uploaded_file, compression='zip', sep=None, engine='python')
                     st.info(f"ℹ️ Detected ZIP-compressed CSV file for {uploaded_file.name}. Successfully decompressed.")
                 except Exception as e_zip:
                     raise Exception(f"Failed as Excel ({e_xls}). Failed as Zipped CSV ({e_zip}).")
             else:
                 raise e_xls
                 
    return df_input

st.set_page_config(page_title="Curadoria QSAR", layout="wide")

# 1. Render Sidebar & Get Config
config = render_sidebar()
t = config['t']

# Routing Logic
if config.get('app_mode') == t.get('mode_prediction', "Prediction (Virtual Screening)"):
    from src.ui.prediction import render_prediction_page
    render_prediction_page(config)
else:
    # --- Standard Curation & Training Mode ---
    
    t = config['t']
    uploaded_files = config['uploaded_file'] # Now a list
    run_btn = config['run_btn']

    # Title & Intro (Main Page)
    # Logo Emphasis
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("assets/logo.png", use_container_width=True)
    
    st.markdown(f"<p style='text-align: center; color: grey;'>{t.get('site_summary', '')}</p>", unsafe_allow_html=True)
        
    st.title(t['title'])
    st.markdown(t['intro_text'])
    
    # Graphical Abstract
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image("assets/graphical_abstract.png", use_container_width=True, caption="ScreenSAR Workflow: From Chaos to Precision")
    
    with st.expander(t['pipeline_expander']):
        st.markdown(t['pipeline_desc'])

    # Initialize session state
    if 'curated_result' not in st.session_state:
        st.session_state.curated_result = None
    if 'input_len' not in st.session_state:
        st.session_state.input_len = 0

    # 2. Main Logic: Execution
    if uploaded_files and run_btn:
        try:
            all_dfs = []
            
            # Progress bar for file loading if multiple files
            files_to_load = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            
            for f in files_to_load:
                try:
                    df = read_single_file(f)
                    if df is not None and not df.empty:
                        # Optional: Add a column to track source file? 
                        # df['Source_File'] = f.name 
                        all_dfs.append(df)
                    else:
                        st.warning(f"File {f.name} resulted in empty DataFrame.")
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")
            
            if not all_dfs:
                st.error("No valid data loaded.")
                st.stop()
                
            # Concatenate
            df_input = pd.concat(all_dfs, ignore_index=True)
                
            st.subheader(t['preview_header'])
            st.write(f"**Total Records Loaded:** {len(df_input)} (merged from {len(all_dfs)} files)")
            st.dataframe(df_input.head())
            
            with st.status(t['status_running'], expanded=True) as status:
                st.write(t['status_init'])
                curador = CuradoriaQSAR(df_input, corte_ativo_nm=config['corte_nm'])
                
                df_result = curador.executar_pipeline(calculate_pIC50=config['calc_pic50'])
                
                st.session_state.curated_result = df_result
                st.session_state.input_len = len(df_input)
                
                # Clear previous modeling results
                if 'modeling_results' in st.session_state: del st.session_state['modeling_results']
                if 'roc_data' in st.session_state: del st.session_state['roc_data']
                if 'trained_models' in st.session_state: del st.session_state['trained_models']
                
                status.update(label=t['status_complete'], state="complete", expanded=False)
                
        except Exception as e:
            st.error(t['error_generic'].format(e))
            st.exception(e)

    elif not uploaded_files:
        # Basic instruction
        st.info("Upload files to start." if config['lang']=="English" else ("Faça upload para começar." if config['lang']=="Português" else "Datei hochladen."))

    # 3. Render Dashboard (Results)
    render_dashboard(config)
