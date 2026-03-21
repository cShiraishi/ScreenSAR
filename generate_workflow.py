from graphviz import Digraph

def create_screensar_flowchart():
    dot = Digraph('ScreenSAR_Workflow', comment='ScreenSAR Platform Workflow')
    dot.attr(rankdir='TB', size='10,10', splines='ortho')
    
    # Global node style
    dot.attr('node', shape='box', style='filled', fillcolor='white', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # --- INPUT ---
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='invis')
        c.node('Input', 'Raw Datasets\n(CSV / Excel)', shape='folder', fillcolor='#E0E0E0')

    # --- CURATION MODULE ---
    with dot.subgraph(name='cluster_curation') as c:
        c.attr(label='1. Data Curation (CuradoriaQSAR)', style='filled', color='#f2f2f2', fontname='Helvetica-Bold')
        c.node('ChemNorm', 'Chemical Normalization\n(Standardize, Desalt, Fragment)')
        c.node('SMILESClean', 'SMILES Canonicalization\n(RDKit)')
        c.node('UnitStd', 'Activity Standardization\n(Converto to nM, pIC50)')
        c.node('DupFilter', 'Duplicate Handling\n(Geometric Mean / Discordance Check)')
        c.node('CuratedData', 'Curated Dataset\n(Clean SMILES + Binary Labels)', shape='note')
        
        c.edge('ChemNorm', 'SMILESClean')
        c.edge('SMILESClean', 'UnitStd')
        c.edge('UnitStd', 'DupFilter')
        c.edge('DupFilter', 'CuratedData')

    # --- MODELING MODULE ---
    with dot.subgraph(name='cluster_modeling') as c:
        c.attr(label='2. QSAR Modeling (ModeladorQSAR)', style='filled', color='#e6f3ff', fontname='Helvetica-Bold')
        c.node('FeatEng', 'Feature Engineering\n(Morgan, MACCS, RDKit Fingerprints)')
        c.node('Split', 'Data Splitting\n(Stratified Train/Test)')
        c.node('AD', 'Applicability Domain (AD)\n(k-NN, Z-score < 3.0)')
        c.node('Training', 'Model Training\n(RF, SVM, GBM, KNN, LR)')
        c.node('Eval', 'Model Evaluation & Ranking\n(MCC, AUC, Sensitivity)', shape='component')
        
        c.edge('FeatEng', 'Split')
        c.edge('Split', 'AD')
        c.edge('AD', 'Training')
        c.edge('Training', 'Eval')

    # --- SCREENING MODULE ---
    with dot.subgraph(name='cluster_screening') as c:
        c.attr(label='3. Virtual Screening & Deployment', style='filled', color='#e6ffe6', fontname='Helvetica-Bold')
        c.node('BestModel', 'Select Best Model\n(Max MCC)')
        c.node('Report', 'Generate PDF Report\n(ROC Curves, Metrics)', shape='note')
        c.node('Screening', 'Virtual Screening\n(New Compounds)', shape='component')
        c.node('Output', 'Prediction Results\n(Probability & Class)', shape='folder', fillcolor='#E0E0E0')
        
        c.edge('BestModel', 'Report')
        c.edge('BestModel', 'Screening')
        c.edge('Screening', 'Output')

    # --- MAIN FLOW EDGES ---
    dot.edge('Input', 'ChemNorm')
    dot.edge('CuratedData', 'FeatEng')
    dot.edge('Eval', 'BestModel')

    # Render
    output_path = 'docs/assets/screensar_workflow'
    dot.render(output_path, format='png', cleanup=True)
    print(f"Flowchart generated at: {output_path}.png")

if __name__ == "__main__":
    create_screensar_flowchart()
