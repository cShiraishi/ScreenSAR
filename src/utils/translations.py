
translations = {
    "Português": {
        "title": "💊 Curadoria de Dados QSAR",
        "site_summary": "🚀 **ScreenSAR:** O futuro da descoberta de fármacos. Curadoria, IA e Triagem Virtual em uma experiência fluida.",
        "intro_text": "Esta ferramenta realiza a curadoria automática de dados do ChEMBL para estudos de QSAR.",
        "pipeline_expander": "ℹ️ Detalhes do Pipeline de Curadoria",
        "pipeline_desc": """
        **1. Entrada e Configuração**
        - Upload do arquivo ChEMBL (CSV/Excel).
        - Definição de cortes (Atividade/pIC50).
        - **Seleção de Descritores:** Escolha do algoritmo (Morgan, MACCS, RDKit).

        **2. Limpeza Química e Normalização**
        - Conversão para SMILES canônico (RDKit).
        - Remoção de sais e solventes.
        - Seleção do maior fragmento orgânico.
        - Remoção de metais e inorgânicos.
        - **Estereoquímica:** Neutralização e remoção de isômeros para consistência 2D.
        
        **3. Tratamento de Atividade e Unidades**
        - Conversão automática de unidades (uM, mM, M) para **nM**.
        - Cálculo de **pIC50** (-log10[M]) se solicitado.
        
        **4. Gestão de Duplicatas**
        - Identificação de compostos idênticos (mesmo SMILES).
        - **Concordantes:** Média geométrica dos valores de atividade.
        - **Discordantes:** Removidos se a variação for maior que 1 log (10x).
        
        **5. Classificação Binária**
        - Definição de **Ativos (1)** e **Inativos (0)** com base no limiar (cutoff) definido (ex: 100 nM ou pIC50 7.0).

        **6. Engenharia de Atributos (Descritores)**
        - Geração de fingerprints moleculares (Morgan, MACCS, RDKit).
        - Vetorização da estrutura química para Machine Learning.
        """,
        "settings": "Configurações",
        "upload_label": "Upload Arquivo ChEMBL (CSV/Excel)",
        "curation_params": "Parâmetros de Curadoria",
        "cutoff_unit": "Unidade de Corte",
        "cutoff_nm_label": "Corte de Atividade (nM)",
        "cutoff_pic50_label": "Corte de Atividade (pIC50/pEC50)",
        "cutoff_pic50_help": "Compostos com pIC50 >= a este valor serão considerados ATIVOS (1).",
        "calc_pic50": "Calcular pIC50 (-log10)",
        "run_btn": "Executar Curadoria",
        "preview_header": "Prévia dos Dados Originais",
        "status_running": "Executando Pipeline...",
        "status_init": "Inicializando curadoria...",
        "status_complete": "Curadoria Concluída!",
        "success_msg": "Sucesso! {} compostos únicos processados.",
        "total_orig": "Total Original",
        "total_final": "Total Final",
        "removed": "Compostos Removidos",
        "curated_header": "Dados Curados",
        "download_csv": "📥 CSV Completo",
        "download_actives": "📥 Ativos (XLSX)",
        "download_inactives": "📥 Inativos (XLSX)",
        "dist_header": "📊 Análise de Distribuição",
        "actives": "Ativos (1)",
        "inactives": "Inativos (0)",
        "active_singular": "Ativo",
        "inactive_singular": "Inativo",
        "outlier_header": "🔍 Análise de Outliers",
        "expander_outlier": "Ver Detalhes dos Outliers",
        "analyzing_dist": "Analisando distribuição de:",
        "mean": "Média",
        "std": "Desvio Padrão",
        "potential_outliers": "Potenciais Outliers (>3σ)",
        "outlier_warning": "Foram encontrados {} compostos muito distantes da média (3σ).",
        "outlier_none": "Dados insuficientes para análise de outliers.",
        "chem_space_header": "🧪 Análise do Espaço Químico",
        "expander_chem_space": "Gerar Visualização do Espaço Químico (PCA)",
        "chem_space_info": "Esta análise gera fingerprints moleculares e usa PCA.",
        "descriptor_label": "Tipo de Descritor",
        "descriptor_help": "Escolha o tipo de fingerprint molecular.",
        "nbits": "Número de Bits",
        "radius": "Raio (Morgan)",
        "gen_map_btn": "Gerar Mapa Químico",
        "map_title": "Mapa do Espaço Químico (PCA)",
        "var_explained": "Variância Explicada",
        "model_header": "🤖 Modelagem QSAR (Machine Learning)",
        "model_expander": "Construir e Avaliar Modelos",
        "model_intro": "Treine modelos preditivos usando os dados curados.",
        "calc_modi": "🔍 Calcular Índice de Modelabilidade (MODI)",
        "modi_spinner": "Calculando MODI...",
        "modi_high": "**Alta Modelabilidade!** (MODI >= 0.65).",
        "modi_low": "**Baixa Modelabilidade.** (MODI < 0.65).",
        "select_models": "Escolha os Modelos",
        "test_split": "Tamanho do Conjunto de Teste (%)",
        "test_split_help": "Porcentagem reservada para teste.",
        "test_split_explanation": "O conjunto de teste é separado para avaliar o modelo em dados nunca vistos. Recomendado: 20%.",
        "train_btn": "Treinar Modelos",
        "warn_select_model": "Selecione pelo menos um modelo.",
        "training_spinner": "Preparando dados e treinando modelos...",
        "training_success": "Treinamento Concluído!",
        "metrics_header": "📊 Métricas de Performance",
        "roc_header": "📈 Curvas ROC",
        "viz_header": "📊 Comparação Visual de Métricas",
        "download_models_header": "💾 Download dos Modelos (.pkl)",
        "download_models_text": "Baixe o arquivo do modelo treinado.",
        "error_insufficient": "Dados insuficientes.",
        "failed_models": "Alguns modelos falharam durante o treinamento:",
        "all_failed": "Todos os modelos falharam.",
        "empty_results": "Resultados vazios.",
        "error_generic": "Ocorreu um erro: {}",
        "error_prefix": "Erro",
        "best_model_header": "🏆 Melhor Modelo Sugerido",
        "best_model_rec": "Com base no **MCC**, o modelo recomendado é: **{}**.",
        "best_model_metrics": "Performance: MCC={:.3f}, Acurácia={:.3f}, F1={:.3f}.",
        "download_report_btn": "📄 Baixar Relatório Completo (PDF)",
        "report_filename": "relatorio_qsar_melhor_modelo.pdf",
        "report_title": "RELATÓRIO DE MODELAGEM QSAR",
        "report_summary_sec": "1. RESUMO E RECOMENDAÇÃO",
        "report_best_model": "Melhor Modelo Sugerido: {}",
        "report_rationale": "Critério de escolha: Maior Índice de Correlação de Matthews (MCC).",
        "report_dataset_sec": "2. INFORMAÇÕES DO DATASET",
        "report_models_sec": "3. PERFORMANCE DOS MODELOS",
        "report_footer": "Gerado automaticamente por QSAR Data Curation Tool.",
        "pred_title": "🎯 Predição / Triagem Virtual",
        "pred_intro": "Use este módulo para triar novos compostos usando um modelo pré-treinado.\n1. **Upload do Modelo**: Carregue o arquivo `.pkl` que você baixou do módulo de Treinamento.\n2. **Upload de Compostos**: Forneça um arquivo CSV ou TXT com os SMILES.",
        "pred_step1": "1. Carregar Modelo Treinado (.pkl)",
        "upload_model_label": "Upload do Modelo",
        "model_loaded": "✅ Carregado: **{}**",
        "model_config": "⚙️ Config: **{}** ({} bits, r={})",
        "legacy_warn": "⚠️ Modelo antigo detectado (sem metadados). Assumindo Morgan 1024, raio 2.",
        "pred_step2": "2. Upload de Compostos para Triagem",
        "upload_mols_label": "Upload CSV/TXT (deve ter coluna 'SMILES' ou ser uma lista)",
        "analyzed_mols": "Analisados {} compostos da coluna `{}`",
        "run_pred_btn": "🚀 Executar Predição",
        "pred_results_title": "Resultados da Predição",
        "pred_summary": "Previstos {} Ativos de {} moléculas.",
        "download_pred": "📥 Baixar Resultados (CSV)",
        "error_no_descriptors": "Não foi possível gerar descritores para nenhuma molécula.",
        "error_smiles_col": "Não foi possível verificar a coluna SMILES. Certifique-se de que o nome da coluna contém 'SMILES'.",
        "report_scaffolds_sec": "3. Principais Escafolds (Ativos)",
        "report_scaffolds_desc": "Os seguintes escafolds são os mais frequentes entre as moléculas previstas como ativas. Eles representam a estrutura básica (core) dos seus melhores hits.",
        "scaffold_freq": "Frequência: ",
        "gen_pdf_header": "📄 Gerar Relatório PDF",
        "gen_pdf_desc": "Gere um relatório PDF resumido destacando os principais hits ativos que estão dentro do Domínio de Aplicabilidade e seus principais escafolds.",
        "sidebar_mode_label": "Modo / Mode",
        "mode_curation": "Treinamento & Curadoria",
        "mode_prediction": "Predição (Triagem Virtual)",
        "recommended": "Recomendado",
        "best_model_tooltip": "Melhor modelo encontrado"
    },
    "English": {
        "title": "💊 QSAR Data Curation",
        "site_summary": "🚀 **ScreenSAR:** The future of drug discovery. Curation, AI, and Virtual Screening in a seamless experience.",
        "intro_text": "This tool performs automatic curation of ChEMBL data for QSAR studies.",
        "pipeline_expander": "ℹ️ Curation Pipeline Details",
        "pipeline_desc": """
        **1. Input & Configuration**
        - File upload (CSV/Excel).
        - Activity cutoff selection.
        - **Descriptor Selection:** Choose algorithm (Morgan, MACCS, RDKit).

        **2. Chemical Cleaning & Normalization**
        - Canonical SMILES conversion (RDKit).
        - Salt and solvent removal.
        - Largest organic fragment selection.
        - Metal/inorganic removal.
        - **Stereochemistry:** Neutralization and removal of isomers for 2D consistency.
        
        **3. Activity Treatment**
        - Automatic unit conversion to **nM**.
        - **pIC50** calculation if requested.
        
        **4. Duplicate Management**
        - Identification of identical compounds.
        - **Concordant:** Geometric mean of activity values.
        - **Discordant:** Removed if variation > 1 log.
        
        **5. Classification**
        - **Active (1)** / **Inactive (0)** based on cutoff.

        **6. Feature Engineering (Descriptors)**
        - Molecular fingerprint generation (Morgan, MACCS, RDKit).
        - Chemical structure vectorization for Machine Learning.
        """,
        "settings": "Settings",
        "upload_label": "Upload ChEMBL File (CSV/Excel)",
        "curation_params": "Curation Parameters",
        "cutoff_unit": "Cutoff Unit",
        "cutoff_nm_label": "Activity Cutoff (nM)",
        "cutoff_pic50_label": "Activity Cutoff (pIC50/pEC50)",
        "cutoff_pic50_help": "Compounds with pIC50 >= this value are ACTIVE (1).",
        "calc_pic50": "Calculate pIC50 (-log10)",
        "run_btn": "Run Curation",
        "preview_header": "Original Data Preview",
        "status_running": "Running Pipeline...",
        "status_init": "Initializing curation...",
        "status_complete": "Curation Complete!",
        "success_msg": "Success! {} unique compounds processed.",
        "total_orig": "Original Total",
        "total_final": "Final Total",
        "removed": "Removed Compounds",
        "curated_header": "Curated Data",
        "download_csv": "📥 Full CSV",
        "download_actives": "📥 Actives (XLSX)",
        "download_inactives": "📥 Inactives (XLSX)",
        "dist_header": "📊 Distribution Analysis",
        "actives": "Actives (1)",
        "inactives": "Inactives (0)",
        "active_singular": "Active",
        "inactive_singular": "Inactive",
        "outlier_header": "🔍 Outlier Analysis",
        "expander_outlier": "View Outlier Details",
        "analyzing_dist": "Analyzing distribution of:",
        "mean": "Mean",
        "std": "Std Dev",
        "potential_outliers": "Potential Outliers (>3σ)",
        "outlier_warning": "Found {} compounds far from mean (3σ).",
        "outlier_none": "Insufficient data for outlier analysis.",
        "chem_space_header": "🧪 Chemical Space Analysis",
        "expander_chem_space": "Generate Chemical Space Visualization (PCA)",
        "chem_space_info": "Generates molecular fingerprints and uses PCA.",
        "nbits": "Number of Bits",
        "radius": "Radius (Morgan)",
        "gen_map_btn": "Generate Chemical Map",
        "map_title": "Chemical Space Map (PCA)",
        "var_explained": "Variance Explained",
        "model_header": "🤖 QSAR Modeling (Machine Learning)",
        "model_expander": "Build and Evaluate Models",
        "model_intro": "Train predictive models using curated data.",
        "calc_modi": "🔍 Calculate Modelability Index (MODI)",
        "modi_spinner": "Calculating MODI...",
        "modi_high": "**High Modelability!** (MODI >= 0.65).",
        "modi_low": "**Low Modelability.** (MODI < 0.65).",
        "select_models": "Choose Models",
        "test_split": "Test Set Size (%)",
        "test_split_help": "Percentage reserved for testing.",
        "test_split_explanation": "The test set is reserved to evaluate the model on unseen data. Recommended: 20%.",
        "train_btn": "Train Models",
        "warn_select_model": "Select at least one model.",
        "training_spinner": "Preparing data and training models...",
        "training_success": "Training Complete!",
        "metrics_header": "📊 Performance Metrics",
        "roc_header": "📈 ROC Curves",
        "viz_header": "📊 Visual Metrics Comparison",
        "download_models_header": "💾 Download Models (.pkl)",
        "download_models_text": "Download trained model file.",
        "error_insufficient": "Insufficient data.",
        "failed_models": "Some models failed during training:",
        "all_failed": "All models failed.",
        "empty_results": "Empty results.",
        "error_generic": "An error occurred: {}",
        "error_prefix": "Error",
        "best_model_header": "🏆 Best Model Suggestion",
        "best_model_rec": "Based on **MCC**, the recommended model is: **{}**.",
        "best_model_metrics": "Performance: MCC={:.3f}, Accuracy={:.3f}, F1={:.3f}.",
        "download_report_btn": "📄 Download Full Report (PDF)",
        "report_filename": "qsar_modeling_report.pdf",
        "report_title": "QSAR MODELING REPORT",
        "report_summary_sec": "1. SUMMARY & RECOMMENDATION",
        "report_best_model": "Best Model Suggestion: {}",
        "report_rationale": "Selection Criterion: Highest Matthews Correlation Coefficient (MCC).",
        "report_dataset_sec": "2. DATASET INFORMATION",
        "report_models_sec": "3. MODEL PERFORMANCE",
        "report_footer": "Generated automatically by QSAR Data Curation Tool.",
        "pred_title": "🎯 Prediction / Virtual Screening",
        "pred_intro": "Use this module to screen new compounds using a pre-trained model.\n1. **Upload Model**: Load the `.pkl` file you downloaded from the Training module.\n2. **Upload Compounds**: Provide a CSV or TXT file with SMILES.",
        "pred_step1": "1. Load Trained Model (.pkl)",
        "upload_model_label": "Upload Model",
        "model_loaded": "✅ Loaded: **{}**",
        "model_config": "⚙️ Config: **{}** ({} bits, r={})",
        "legacy_warn": "⚠️ Legacy model detected (no metadata). Assuming Morgan 1024, radius 2.",
        "pred_step2": "2. Upload Compounds for Screening",
        "upload_mols_label": "Upload CSV/TXT (must have 'SMILES' column or be a list)",
        "analyzed_mols": "Analyzed {} compounds from column `{}`",
        "run_pred_btn": "🚀 Run Prediction",
        "pred_results_title": "Prediction Results",
        "pred_summary": "Predicted {} Actives out of {} molecules.",
        "download_pred": "📥 Download Results (CSV)",
        "error_no_descriptors": "Could not generate descriptors for any molecule.",
        "error_smiles_col": "Could not verify SMILES column. Please ensure column name contains 'SMILES'.",
        "report_scaffolds_sec": "3. Principal Scaffolds (Active Hits)",
        "report_scaffolds_desc": "The following scaffolds are the most frequent among the predicted active molecules. These represent the core structural frameworks of your top hits.",
        "scaffold_freq": "Frequency: ",
        "gen_pdf_header": "📄 Generate PDF Report",
        "gen_pdf_desc": "Generate a summary PDF report highlighting the top Active hits that fall Inside the Applicability Domain and their principal scaffolds.",
        "sidebar_mode_label": "Mode",
        "mode_curation": "Training/Curation",
        "mode_prediction": "Prediction (Virtual Screening)",
        "recommended": "Recommended",
        "best_model_tooltip": "Best model found"
    },
    "Deutsch": {
        "title": "💊 QSAR-Datenkuration",
        "site_summary": "🚀 **ScreenSAR:** Die Zukunft der Wirkstoffforschung. Kuration, KI und Virtuelles Screening in einer nahtlosen Erfahrung.",
        "intro_text": "Dieses Tool führt eine automatische Kuration von ChEMBL-Daten durch.",
        "pipeline_expander": "ℹ️ Details zur Kurations-Pipeline",
        "pipeline_desc": """
        **1. Eingabe & Konfiguration**
        - Datei-Upload (CSV/Excel).
        - Auswahl des Aktivitätsgrenzwerts.
        - **Deskriptor-Auswahl:** Algorithmus wählen (Morgan, MACCS, RDKit).

        **2. Chemische Reinigung**
        - SMILES-Konvertierung (RDKit).
        - Entfernung von Salzen/Lösungsmitteln.
        - **Stereochemie:** Neutralisierung und Entfernung von Isomeren.
        
        **3. Aktivität**
        - Umrechnung in **nM**.
        - Berechnung von **pIC50**.
        
        **4. Duplikate**
        - Geometrischer Mittelwert oder Entfernung.
        
        **5. Klassifizierung**
        - **Aktiv (1)** / **Inaktiv (0)**.

        **6. Feature Engineering (Deskriptoren)**
        - Fingerprint-Generierung (Morgan, MACCS, RDKit).
        - Vektorisierung für Machine Learning.
        """,
        "settings": "Einstellungen",
        "upload_label": "ChEMBL-Datei hochladen (CSV/Excel)",
        "curation_params": "Kurationsparameter",
        "cutoff_unit": "Grenzwert-Einheit",
        "cutoff_nm_label": "Aktivitätsgrenzwert (nM)",
        "cutoff_pic50_label": "Aktivitätsgrenzwert (pIC50/pEC50)",
        "cutoff_pic50_help": "Verbindungen mit pIC50 >= Wert sind AKTIV (1).",
        "calc_pic50": "Berechne pIC50 (-log10)",
        "run_btn": "Kuration ausführen",
        "preview_header": "Vorschau",
        "status_running": "Pipeline läuft...",
        "status_init": "Initialisierung...",
        "status_complete": "Kuration fertig!",
        "success_msg": "Erfolg! {} Verbindungen verarbeitet.",
        "total_orig": "Original",
        "total_final": "Endgültig",
        "removed": "Entfernt",
        "curated_header": "Kuratierte Daten",
        "download_csv": "📥 CSV",
        "download_actives": "📥 Aktive (XLSX)",
        "download_inactives": "📥 Inaktive (XLSX)",
        "dist_header": "📊 Verteilung",
        "actives": "Aktiv (1)",
        "inactives": "Inaktiv (0)",
        "active_singular": "Aktiv",
        "inactive_singular": "Inaktiv",
        "outlier_header": "🔍 Ausreißer",
        "expander_outlier": "Details anzeigen",
        "analyzing_dist": "Analyse von:",
        "mean": "Mittelwert",
        "std": "StdAbw",
        "potential_outliers": "Potenzielle Ausreißer (>3σ)",
        "outlier_warning": "{} Ausreißer gefunden.",
        "outlier_none": "Zu wenig Daten.",
        "chem_space_header": "🧪 Chemischer Raum",
        "expander_chem_space": "Visualisierung generieren (PCA)",
        "chem_space_info": "Generiert Fingerprints und nutzt PCA.",
        "nbits": "Bits",
        "radius": "Radius",
        "gen_map_btn": "Karte generieren",
        "map_title": "Chemischer Raum (PCA)",
        "var_explained": "Erklärte Varianz",
        "model_header": "🤖 QSAR-Modellierung",
        "model_expander": "Modelle erstellen",
        "model_intro": "Modelle trainieren mit kuratierten Daten.",
        "calc_modi": "🔍 MODI berechnen",
        "modi_spinner": "MODI berechnen...",
        "modi_high": "**Hoch!** (MODI >= 0.65).",
        "modi_low": "**Niedrig.** (MODI < 0.65).",
        "select_models": "Modelle wählen",
        "test_split": "Test-Größe (%)",
        "test_split_help": "% für Test reserviert.",
        "test_split_explanation": "Der Testdatensatz dient zur Bewertung an unbekannten Daten. Empfohlen: 20%.",
        "train_btn": "Training starten",
        "warn_select_model": "Bitte Modell wählen.",
        "training_spinner": "Training läuft...",
        "training_success": "Training fertig!",
        "metrics_header": "📊 Metriken",
        "roc_header": "📈 ROC-Kurven",
        "viz_header": "📊 Visueller Vergleich",
        "download_models_header": "💾 Modelle laden (.pkl)",
        "download_models_text": "Modell zur Wiederverwendung speichern.",
        "error_insufficient": "Zu wenig Daten.",
        "failed_models": "Fehler bei einigen Modellen:",
        "all_failed": "Alle Modelle fehlgeschlagen.",
        "empty_results": "Keine Ergebnisse.",
        "error_generic": "Ein Fehler ist aufgetreten: {}",
        "error_prefix": "Fehler",
        "best_model_header": "🏆 Bester Modellvorschlag",
        "best_model_rec": "Basierend auf **MCC** ist das empfohlene Modell: **{}**.",
        "best_model_metrics": "Leistung: MCC={:.3f}, Genauigkeit={:.3f}, F1={:.3f}.",
        "download_report_btn": "📄 Vollständigen Bericht herunterladen (PDF)",
        "report_filename": "qsar_modellierung_bericht.pdf",
        "report_title": "QSAR-MODELLIERUNGSBERICHT",
        "report_summary_sec": "1. ZUSAMMENFASSUNG & EMPFEHLUNG",
        "report_best_model": "Bestes Modell: {}",
        "report_rationale": "Auswahlkriterium: Höchster Matthews-Korrelationskoeffizient (MCC).",
        "report_dataset_sec": "2. DATASET-INFORMATIONEN",
        "report_models_sec": "3. MODELLLEISTUNG",
        "report_footer": "Automatisch generiert vom QSAR Data Curation Tool.",
        "pred_title": "🎯 Vorhersage / Virtuelles Screening",
        "pred_intro": "Nutzen Sie dieses Modul, um neue Verbindungen zu screenen.\n1. **Modell laden**: Laden Sie die `.pkl`-Datei.\n2. **Verbindungen laden**: CSV oder TXT mit SMILES bereitstellen.",
        "pred_step1": "1. Trainiertes Modell laden (.pkl)",
        "upload_model_label": "Modell hochladen",
        "model_loaded": "✅ Geladen: **{}**",
        "model_config": "⚙️ Konfig: **{}** ({} Bits, r={})",
        "legacy_warn": "⚠️ Altes Modell erkannt (keine Metadaten). Angenommen Morgan 1024.",
        "pred_step2": "2. Verbindungen für Screening laden",
        "upload_mols_label": "CSV/TXT hochladen (muss 'SMILES'-Spalte enthalten)",
        "analyzed_mols": "Analysiert: {} Verbindungen aus Spalte `{}`",
        "run_pred_btn": "🚀 Vorhersage starten",
        "pred_results_title": "Vorhersageergebnisse",
        "pred_summary": "Vorhergesagt: {} Aktive von {} Molekülen.",
        "download_pred": "📥 Ergebnisse herunterladen (CSV)",
        "error_no_descriptors": "Konnte keine Deskriptoren generieren.",
        "error_smiles_col": "SMILES-Spalte nicht gefunden.",
        "sidebar_mode_label": "Modus",
        "mode_curation": "Training/Kuration",
        "mode_prediction": "Vorhersage (Virtuelles Screening)",
        "recommended": "Empfohlen",
        "best_model_tooltip": "Bestes gefundenes Modell"
    },
    "中文": {
        "title": "💊 QSAR 数据整理",
        "site_summary": "🚀 **ScreenSAR:** 药物发现的未来。在一个无缝体验中实现整理、AI 和虚拟筛选。",
        "intro_text": "该工具对 ChEMBL 数据进行自动整理以用于 QSAR 研究。",
        "pipeline_expander": "ℹ️ 整理流程详情",
        "pipeline_desc": """
        **1. 输入与配置**
        - 上传 ChEMBL 文件 (CSV/Excel)。
        - 选择活性截止值。
        - **描述符选择：** 选择算法 (Morgan, MACCS, RDKit)。

        **2. 化学清洗与标准化**
        - 规范 SMILES 转换 (RDKit)。
        - 去除盐和溶剂。
        - 选择最大有机片段。
        - 去除金属和无机物。
        - **立体化学：** 中和并去除异构体以保持 2D 一致性。
        
        **3. 活性处理与单位**
        - 自动将单位 (uM, mM, M) 转换为 **nM**。
        - 如果请求，计算 **pIC50** (-log10[M])。
        
        **4. 重复项管理**
        - 识别相同化合物 (相同 SMILES)。
        - **一致：** 活性值的几何平均值。
        - **不一致：** 如果变化 > 1 log (10倍) 则移除。
        
        **5. 二元分类**
        - 基于设定的截止值定义 **活性 (1)** 和 **非活性 (0)** (例如: 100 nM 或 pIC50 7.0)。

        **6. 特征工程 (描述符)**
        - 分子指纹生成 (Morgan, MACCS, RDKit)。
        - 机器学习的化学结构向量化。
        """,
        "settings": "设置",
        "upload_label": "上传 ChEMBL 文件 (CSV/Excel)",
        "curation_params": "整理参数",
        "cutoff_unit": "截止单位",
        "cutoff_nm_label": "活性截止值 (nM)",
        "cutoff_pic50_label": "活性截止值 (pIC50/pEC50)",
        "cutoff_pic50_help": "pIC50 >= 此值的化合物被视为活性 (1)。",
        "calc_pic50": "计算 pIC50 (-log10)",
        "run_btn": "运行整理",
        "preview_header": "原始数据预览",
        "status_running": "正在运行流程...",
        "status_init": "正在初始化整理...",
        "status_complete": "整理完成！",
        "success_msg": "成功！处理了 {} 个唯一化合物。",
        "total_orig": "原始总数",
        "total_final": "最终总数",
        "removed": "已移除化合物",
        "curated_header": "已整理数据",
        "download_csv": "📥 完整 CSV",
        "download_actives": "📥 活性 (XLSX)",
        "download_inactives": "📥 非活性 (XLSX)",
        "dist_header": "📊 分布分析",
        "actives": "活性 (1)",
        "inactives": "非活性 (0)",
        "active_singular": "活性",
        "inactive_singular": "非活性",
        "outlier_header": "🔍 异常值分析",
        "expander_outlier": "查看异常值详情",
        "analyzing_dist": "正在分析分布：",
        "mean": "平均值",
        "std": "标准差",
        "potential_outliers": "潜在异常值 (>3σ)",
        "outlier_warning": "发现 {} 个远离平均值 (3σ) 的化合物。",
        "outlier_none": "数据不足以进行异常值分析。",
        "chem_space_header": "🧪 化学空间分析",
        "expander_chem_space": "生成化学空间可视化 (PCA)",
        "chem_space_info": "生成分子指纹并使用 PCA。",
        "descriptor_label": "描述符类型",
        "descriptor_help": "选择分子指纹类型。",
        "nbits": "位数",
        "radius": "半径 (Morgan)",
        "gen_map_btn": "生成化学地图",
        "map_title": "化学空间地图 (PCA)",
        "var_explained": "解释方差",
        "model_header": "🤖 QSAR 建模 (机器学习)",
        "model_expander": "构建和评估模型",
        "model_intro": "使用已整理的数据训练预测模型。",
        "calc_modi": "🔍 计算可建模性指数 (MODI)",
        "modi_spinner": "正在计算 MODI...",
        "modi_high": "**高可建模性！** (MODI >= 0.65)。",
        "modi_low": "**低可建模性。** (MODI < 0.65)。",
        "select_models": "选择模型",
        "test_split": "测试集大小 (%)",
        "test_split_help": "保留用于测试的百分比。",
        "test_split_explanation": "测试集用于评估模型在未见数据上的表现。推荐：20%。",
        "train_btn": "训练模型",
        "warn_select_model": "至少选择一个模型。",
        "training_spinner": "正在准备数据并训练模型...",
        "training_success": "训练完成！",
        "metrics_header": "📊 性能指标",
        "roc_header": "📈 ROC 曲线",
        "viz_header": "📊 可视化指标比较",
        "download_models_header": "💾 下载模型 (.pkl)",
        "download_models_text": "下载已训练的模型文件。",
        "error_insufficient": "数据不足。",
        "failed_models": "部分模型训练失败：",
        "all_failed": "所有模型失败。",
        "empty_results": "结果为空。",
        "error_generic": "发生错误：{}",
        "error_prefix": "错误",
        "best_model_header": "🏆 最佳模型建议",
        "best_model_rec": "基于 **MCC**，推荐的模型是：**{}**。",
        "best_model_metrics": "性能：MCC={:.3f}, 准确率={:.3f}, F1={:.3f}。",
        "download_report_btn": "📄 下载完整报告 (PDF)",
        "report_filename": "qsar_modeling_report.pdf",
        "report_title": "QSAR 建模报告",
        "report_summary_sec": "1. 总结与建议",
        "report_best_model": "最佳模型建议：{}",
        "report_rationale": "选择标准：最高马修斯相关系数 (MCC)。",
        "report_dataset_sec": "2. 数据集信息",
        "report_models_sec": "3. 模型性能",
        "report_footer": "由 QSAR 数据整理工具自动生成。",
        "pred_title": "🎯 预测 / 虚拟筛选",
        "pred_intro": "使用此模块使用预训练模型筛选新化合物。\n1. **上传模型**：加载从训练模块下载的 `.pkl` 文件。\n2. **上传化合物**：提供带有 SMILES 的 CSV 或 TXT 文件。",
        "pred_step1": "1. 加载已训练模型 (.pkl)",
        "upload_model_label": "上传模型",
        "model_loaded": "✅ 已加载：**{}**",
        "model_config": "⚙️ 配置：**{}** ({} bits, r={})",
        "legacy_warn": "⚠️ 检测到旧模型（无元数据）。假设为 Morgan 1024，半径 2。",
        "pred_step2": "2. 上传化合物进行筛选",
        "upload_mols_label": "上传 CSV/TXT (必须包含 'SMILES' 列或为列表)",
        "analyzed_mols": "分析了列 `{}` 中的 {} 个化合物",
        "run_pred_btn": "🚀 运行预测",
        "pred_results_title": "预测结果",
        "pred_summary": "预测 {} 个分子中有 {} 个活性。",
        "download_pred": "📥 下载结果 (CSV)",
        "error_no_descriptors": "无法为任何分子生成描述符。",
        "error_smiles_col": "无法验证 SMILES 列。请确保列名包含 'SMILES'。",
        "sidebar_mode_label": "模式",
        "mode_curation": "训练/整理",
        "mode_prediction": "预测 (虚拟筛选)",
        "recommended": "推荐",
        "best_model_tooltip": "找到的最佳模型"
    },
    "日本語": {
        "title": "💊 QSARデータキュレーション",
        "site_summary": "🚀 **ScreenSAR:** 創薬の未来。化学キュレーション、AI、そしてバーチャルスクリーニングをシームレスに。",
        "intro_text": "このツールはQSAR研究のためのChEMBLデータの自動キュレーションを実行します。",
        "pipeline_expander": "ℹ️ キュレーションパイプライン詳細",
        "pipeline_desc": """
        **1. 入力と設定**
        - ChEMBLファイルのアップロード (CSV/Excel)。
        - 活性カットオフの選択。
        - **記述子の選択:** アルゴリズムを選択 (Morgan, MACCS, RDKit)。

        **2. 化学的クリーニングと正規化**
        - 標準SMILES変換 (RDKit)。
        - 塩と溶媒の除去。
        - 最大有機フラグメントの選択。
        - 金属と無機物の除去。
        - **立体化学:** 2D整合性のための異性体の中和と除去。
        
        **3. 活性処理と単位**
        - 単位 (uM, mM, M) を **nM** に自動変換。
        - リクエストに応じて **pIC50** (-log10[M]) を計算。
        
        **4. 重複管理**
        - 同一化合物の識別 (同一SMILES)。
        - **一致:** 活性値の幾何平均。
        - **不一致:** 変動 > 1 log (10倍) の場合削除。
        
        **5. 二値分類**
        - 設定されたカットオフに基づいて **活性 (1)** と **不活性 (0)** を定義 (例: 100 nM または pIC50 7.0)。

        **6. 特徴エンジニアリング (記述子)**
        - 分子指紋生成 (Morgan, MACCS, RDKit)。
        - 機械学習のための化学構造ベクトル化。
        """,
        "settings": "設定",
        "upload_label": "ChEMBLファイルのアップロード (CSV/Excel)",
        "curation_params": "キュレーションパラメータ",
        "cutoff_unit": "カットオフ単位",
        "cutoff_nm_label": "活性カットオフ (nM)",
        "cutoff_pic50_label": "活性カットオフ (pIC50/pEC50)",
        "cutoff_pic50_help": "pIC50 >= この値の化合物は活性 (1) と見なされます。",
        "calc_pic50": "pIC50 (-log10) を計算",
        "run_btn": "キュレーション実行",
        "preview_header": "元データプレビュー",
        "status_running": "パイプライン実行中...",
        "status_init": "キュレーション初期化中...",
        "status_complete": "キュレーション完了！",
        "success_msg": "成功！ {} のユニークな化合物が処理されました。",
        "total_orig": "元の合計",
        "total_final": "最終合計",
        "removed": "削除された化合物",
        "curated_header": "キュレーション済みデータ",
        "download_csv": "📥 完全なCSV",
        "download_actives": "📥 活性 (XLSX)",
        "download_inactives": "📥 不活性 (XLSX)",
        "dist_header": "📊 分布分析",
        "actives": "活性 (1)",
        "inactives": "不活性 (0)",
        "active_singular": "活性",
        "inactive_singular": "不活性",
        "outlier_header": "🔍 外れ値分析",
        "expander_outlier": "外れ値の詳細を表示",
        "analyzing_dist": "分布分析中:",
        "mean": "平均",
        "std": "標準偏差",
        "potential_outliers": "潜在的な外れ値 (>3σ)",
        "outlier_warning": "平均から遠く離れた (3σ) {} 個の化合物が見つかりました。",
        "outlier_none": "外れ値分析のためのデータが不足しています。",
        "chem_space_header": "🧪 化学空間分析",
        "expander_chem_space": "化学空間可視化の生成 (PCA)",
        "chem_space_info": "分子指紋を生成しPCAを使用します。",
        "descriptor_label": "記述子タイプ",
        "descriptor_help": "分子指紋タイプを選択。",
        "nbits": "ビット数",
        "radius": "半径 (Morgan)",
        "gen_map_btn": "化学マップ生成",
        "map_title": "化学空間マップ (PCA)",
        "var_explained": "説明分散",
        "model_header": "🤖 QSARモデリング (機械学習)",
        "model_expander": "モデルの構築と評価",
        "model_intro": "キュレーション済みデータを使用して予測モデルをトレーニングします。",
        "calc_modi": "🔍 モデラビリティ指数 (MODI) を計算",
        "modi_spinner": "MODI計算中...",
        "modi_high": "**高モデラビリティ！** (MODI >= 0.65)。",
        "modi_low": "**低モデラビリティ。** (MODI < 0.65)。",
        "select_models": "モデル選択",
        "test_split": "テストセットサイズ (%)",
        "test_split_help": "テスト用に予約された割合。",
        "test_split_explanation": "テストセットは、未知のデータでモデルを評価するために予約されています。推奨: 20%。",
        "train_btn": "モデルをトレーニング",
        "warn_select_model": "少なくとも1つのモデルを選択してください。",
        "training_spinner": "データを準備しモデルをトレーニング中...",
        "training_success": "トレーニング完了！",
        "metrics_header": "📊 パフォーマンス指標",
        "roc_header": "📈 ROC曲線",
        "viz_header": "📊 指標の視覚的比較",
        "download_models_header": "💾 モデルのダウンロード (.pkl)",
        "download_models_text": "トレーニング済みモデルファイルをダウンロード。",
        "error_insufficient": "データ不足。",
        "failed_models": "一部のモデルが失敗しました:",
        "all_failed": "すべてのモデルが失敗しました。",
        "empty_results": "結果は空です。",
        "error_generic": "エラーが発生しました：{}",
        "error_prefix": "エラー",
        "best_model_header": "🏆 最適モデルの提案",
        "best_model_rec": "**MCC** に基づく推奨モデルは: **{}** です。",
        "best_model_metrics": "パフォーマンス: MCC={:.3f}, 正解率={:.3f}, F1={:.3f}。",
        "download_report_btn": "📄 完全なレポートをダウンロード (PDF)",
        "report_filename": "qsar_modeling_report.pdf",
        "report_title": "QSARモデリングレポート",
        "report_summary_sec": "1. 要約と推奨事項",
        "report_best_model": "最適モデル提案: {}",
        "report_rationale": "選択基準: 最高マシューズ相関係数 (MCC)。",
        "report_dataset_sec": "2. データセット情報",
        "report_models_sec": "3. モデルパフォーマンス",
        "report_footer": "QSARデータキュレーションツールによって自動生成。",
        "pred_title": "🎯 予測 / バーチャルスクリーニング",
        "pred_intro": "このモジュールを使用して、事前トレーニング済みモデルを使用して新しい化合物をスクリーニングします。\n1. **モデルのアップロード**: トレーニングモジュールからダウンロードした `.pkl` ファイルをロードします。\n2. **化合物のアップロード**: SMILESを含むCSVまたはTXTファイルを提供します。",
        "pred_step1": "1. トレーニング済みモデルのロード (.pkl)",
        "upload_model_label": "モデルのアップロード",
        "model_loaded": "✅ ロード済み: **{}**",
        "model_config": "⚙️ 設定: **{}** ({} bits, r={})",
        "legacy_warn": "⚠️ 旧モデル検出 (メタデータなし)。Morgan 1024, 半径 2 と仮定。",
        "pred_step2": "2. スクリーニング用化合物のアップロード",
        "upload_mols_label": "CSV/TXTのアップロード ('SMILES'列またはリストが必要)",
        "analyzed_mols": "列 `{}` から {} 個の化合物を分析しました",
        "run_pred_btn": "🚀 予測実行",
        "pred_results_title": "予測結果",
        "pred_summary": "{} 分子中 {} 個の活性を予測。",
        "download_pred": "📥 結果ダウンロード (CSV)",
        "error_no_descriptors": "どの分子に対しても記述子を生成できませんでした。",
        "error_smiles_col": "SMILES列を確認できませんでした。列名に 'SMILES' が含まれていることを確認してください。",
        "sidebar_mode_label": "モード",
        "mode_curation": "トレーニング/キュレーション",
        "mode_prediction": "予測 (バーチャルスクリーニング)",
        "recommended": "推奨",
        "best_model_tooltip": "見つかった最適なモデル"
    }
}
