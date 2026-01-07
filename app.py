import streamlit as st
import pandas as pd
from src.core.curation import CuradoriaQSAR
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard

st.set_page_config(page_title="Curadoria QSAR", layout="wide")

# 1. Render Sidebar & Get Config
config = render_sidebar()

t = config['t']
uploaded_file = config['uploaded_file']
run_btn = config['run_btn']

# Title & Intro (Main Page)
st.title(t['title'])
st.markdown(t['intro_text'])

with st.expander(t['pipeline_expander']):
    st.markdown(t['pipeline_desc'])

# Initialize session state
if 'curated_result' not in st.session_state:
    st.session_state.curated_result = None
if 'input_len' not in st.session_state:
    st.session_state.input_len = 0

# 2. Main Logic: Execution
if uploaded_file and run_btn:
    try:
        # Load data based on file type
        if uploaded_file.name.endswith('.csv'):
            try:
                df_input = pd.read_csv(uploaded_file, sep=';')
                if df_input.shape[1] <= 1:
                    uploaded_file.seek(0)
                    df_input = pd.read_csv(uploaded_file, sep=',')
            except Exception:
                uploaded_file.seek(0)
                df_input = pd.read_csv(uploaded_file, sep=',')
        else:
            df_input = pd.read_excel(uploaded_file)
            
        st.subheader(t['preview_header'])
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

elif not uploaded_file:
    # Basic instruction
    st.info("Upload file to start." if config['lang']=="English" else ("Faça upload para começar." if config['lang']=="Português" else "Datei hochladen."))

# 3. Render Dashboard (Results)
render_dashboard(config)
