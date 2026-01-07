import streamlit as st
from src.utils.translations import translations

def render_sidebar():
    """
    Renders the sidebar and returns the configuration dictionary.
    """
    # Language Selection
    with st.sidebar:
        st.header("Language / Idioma / Sprache")
        lang = st.selectbox("Select Language", ["Português", "English", "Deutsch"], label_visibility="collapsed")
    
    t = translations[lang]
    
    config = {
        "lang": lang,
        "t": t,
        "uploaded_file": None,
        "run_btn": False,
        "corte_nm": 100.0,
        "calc_pic50": True
    }

    # Sidebar Config
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        st.header(t['settings'])
        uploaded_file = st.file_uploader(t['upload_label'], type=['csv', 'xlsx', 'xls'])
        config["uploaded_file"] = uploaded_file
        
        st.subheader(t['curation_params'])
        
        # Unit choice
        cutoff_mode = st.radio(t['cutoff_unit'], ["nM", "pIC50"], horizontal=True)
        
        if cutoff_mode == "nM":
            corte_input = st.number_input(
                t['cutoff_nm_label'], 
                min_value=0.1, 
                value=100.0, 
                help="IC50/EC50 <= X -> Active (1)"
            )
            config["corte_nm"] = corte_input
        else:
            corte_input = st.number_input(
                t['cutoff_pic50_label'], 
                min_value=1.0, 
                max_value=12.0, 
                value=7.0, 
                help=t['cutoff_pic50_help']
            )
            # Convert pIC50 to nM: nM = 10^(9 - pIC50)
            config["corte_nm"] = 10**(9 - corte_input)
            st.caption(f"Equivalent: {config['corte_nm']:.4f} nM")
    
        config["calc_pic50"] = st.checkbox(t['calc_pic50'], value=True)
        
        config["run_btn"] = st.button(t['run_btn'], type="primary", disabled=not uploaded_file)
        
    return config
