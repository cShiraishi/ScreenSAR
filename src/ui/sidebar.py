import streamlit as st
from src.utils.translations import translations

def render_sidebar():
    """
    Renders the sidebar and returns the configuration dictionary.
    """
    # Language Selection
    with st.sidebar:
        st.header("Language / Idioma / Sprache")
        lang = st.selectbox("Select Language", ["Português", "English", "Deutsch", "中文", "日本語"], label_visibility="collapsed")
    
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
        st.image("assets/logo.png", use_container_width=True)
        
        # Mode Selection
        app_mode = st.radio(
            t.get('sidebar_mode_label', 'Mode'), 
            [t.get('mode_curation', "Training/Curation"), t.get('mode_prediction', "Prediction (Virtual Screening)")],
            index=0
        )
        config['app_mode'] = app_mode
        # Compare against the expected values for logic (can be tricky if translated, best to store internal key or check translation)
        # To avoid logic issues, we check if it matches the prediction string
        if app_mode == t.get('mode_prediction', "Prediction (Virtual Screening)"):
             return config # Return early
        
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
        
        st.subheader("Molecular Descriptors")
        descriptor_type = st.selectbox(
            t.get('descriptor_label', 'Descriptor Type'), 
            ["Morgan", "MACCS", "RDKit", "All"],
            index=0,
            help=t.get('descriptor_help', 'Choose fingerprint type.')
        )
        config['descriptor_type'] = descriptor_type
        
        # Only show Morgan params if Morgan is selected
        if descriptor_type == "Morgan":
             config['n_bits'] = st.selectbox(t['nbits'], [1024, 2048, 512], index=0, key='sb_nbits')
             config['radius'] = st.number_input(t['radius'], min_value=1, max_value=4, value=2, key='sb_radius')
        else:
             config['n_bits'] = 1024 # Default/Ignored
             config['radius'] = 2   # Default/Ignored

        config["run_btn"] = st.button(t['run_btn'], type="primary", disabled=not uploaded_file)
        
    # Sidebar Footer with Visitor Counter
    with st.sidebar:
        st.divider()
        st.caption("🌍 Visitor Analytics")
        # Use a generic Flag Counter image. User should replace 'njam' with their own ID from flagcounter.com if needed.
        # This is a free widget that tracks based on the image load.
        st.markdown(
            """
            <a href="https://info.flagcounter.com/njam"><img src="https://s11.flagcounter.com/count/njam/bg_FFFFFF/txt_000000/border_CCCCCC/columns_1/maxflags_10/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
            """,
            unsafe_allow_html=True
        )

    return config
