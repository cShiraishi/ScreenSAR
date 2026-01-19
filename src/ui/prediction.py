import streamlit as st
import pandas as pd
import pickle
import numpy as np
from src.core.curation import CuradoriaQSAR

def render_prediction_page(config):
    t = config['t']
    
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
                     df_mols = pd.read_csv(uploaded_mols, sep=None, engine='python')
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
                    st.write(t['analyzed_mols'].format(len(df_mols), smiles_col))
                    
                    if st.button(t['run_pred_btn']):
                        # Prepare data
                        meta = model_data['meta']
                        
                        # Reuse CuradoriaQSAR for cleaning and fingerprint generation logic
                        # We create a dummy df for curation util
                        df_process = df_mols.copy()
                        df_process['SMILES_Clean'] = df_process[smiles_col] # Simple copy, valid logic handles cleaning
                        
                        # Using Curation class to generate fingerprints
                        # Note: We need to instantiate it. 
                        curator = CuradoriaQSAR(pd.DataFrame({'Smiles': []})) # Dummy init
                        
                        # Manually generating fingerprints to ensure we match the meta
                        fps = []
                        valid_indices = []
                        valid_smiles = []
                        
                        # Clean SMILES first (reuse curation logic if accessible or basic one)
                        # We'll use the one in CuradoriaQSAR if possible, but let's just do direct generation for now to be safe
                        from rdkit import Chem
                        from rdkit.Chem import AllChem, MACCSkeys
                        
                        progress_bar = st.progress(0)
                        
                        # Optimization: Batch processing not implemented here, simple loop
                        count = 0
                        total = len(df_mols)
                        
                        for idx, row in df_process.iterrows():
                            smi = row[smiles_col]
                            if pd.isna(smi): continue
                            
                            try:
                                mol = Chem.MolFromSmiles(str(smi))
                                if mol:
                                    # Generate Descriptor based on Meta
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
                                    valid_indices.append(idx)
                                    valid_smiles.append(smi)
                            except:
                                pass
                            
                            if idx % 100 == 0:
                                progress_bar.progress(min(idx/total, 1.0))
                                
                        progress_bar.progress(1.0)
                        
                        if fps:
                            X_pred = np.array(fps)
                            y_pred = model_data['model'].predict(X_pred)
                            y_proba = model_data['model'].predict_proba(X_pred)[:, 1] if hasattr(model_data['model'], 'predict_proba') else [0]*len(y_pred)
                            
                            # Create Result DF
                            df_res = df_mols.iloc[valid_indices].copy()
                            df_res['Predicted_Class'] = y_pred
                            df_res['Probability_Active'] = y_proba
                            df_res['Prediction_Label'] = ["Active" if x==1 else "Inactive" for x in y_pred]
                            
                            st.subheader(t['pred_results_title'])
                            st.write(t['pred_summary'].format(sum(y_pred), len(y_pred)))
                            
                            st.dataframe(df_res.head())
                            
                            # Download
                            csv = df_res.to_csv(index=False).encode('utf-8')
                            st.download_button(t['download_pred'], csv, "prediction_results.csv", "text/csv")
                            
                        else:
                            st.error(t['error_no_descriptors'])
                        
                else:
                    st.error(t['error_smiles_col'])
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
