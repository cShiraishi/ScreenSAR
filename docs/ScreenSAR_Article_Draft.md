### **Abstract**
The reliability of Quantitative Structure-Activity Relationship (QSAR) modeling is fundamentally dependent on the quality of input data and the robustness of the modeling workflow. This article presents **ScreenSAR**, an automated, end-to-end platform designed to streamline chemical data curation, machine learning model development, and high-throughput virtual screening. ScreenSAR addresses the technical barriers often encountered by medicinal chemists by providing a user-friendly, multilingual interface that automates complex chemoinformatics tasks. The software integrates rigorous data normalization, duplicate management, and multi-algorithm model training with automated performance ranking. A key feature is its chunk-based processing engine, which enables the screening of millions of compounds on standard commercial hardware by optimizing memory usage. By democratizing access to high-quality QSAR workflows, ScreenSAR enhances reproducibility and efficiency in the early stages of drug discovery.

### **Keywords**
QSAR, Virtual Screening, Data Curation, Machine Learning, Chemoinformatics, Automated Pipeline, RDKit, Streamlit, Drug Discovery.

### **Introduction**
In the era of data-driven drug discovery, the integration of high-throughput screening data and public repositories such as ChEMBL and PubChem has enabled the development of predictive models that can significantly accelerate lead optimization. However, the "Garbage In, Garbage Out" principle remains a critical bottleneck; it is estimated that up to 80% of a data scientist's time is spent on data cleaning rather than modeling. Inconsistent chemical representations, stereochemical ambiguities, and experimental noise frequently lead to unreliable models with poor prospective predictive power. Furthermore, most advanced chemoinformatics tools require high-level programming expertise, creating a gap between data science and medicinal chemistry. ScreenSAR was developed to bridge this gap, providing a transparent, automated, and accessible environment for generating reliable QSAR models and deploying them for large-scale virtual screening.

### **Implementation**
ScreenSAR is implemented in Python using the Streamlit framework for the user interface and RDKit for chemoinformatics operations. Its architecture is modular, separating scientific logic from the interface:
*   **Data Curation (`src/core/curation.py`)**: Automates chemical normalization (salt removal, canonicalization) and activity standardization (unit conversion to nM/pIC50). It handles biological duplicates using the geometric mean for concordant entries and discarding discordant ones.
*   **Modeling (`src/core/modeling.py`)**: Supports multiple algorithms (Random Forest, SVM, XGBoost, etc.) and provides automated validation via stratified splitting and metrics such as the Matthews Correlation Coefficient (MCC). It includes the Modelability Index (MODI) to assess dataset quality.
*   **Applicability Domain (`src/core/applicability_domain.py`)**: Implements a k-nearest neighbors (k-NN) approach to define the chemical space and ensure that predictions are only made for molecules similar to the training set (Z-score < 3.0).

**Critical Issue Addressed:** A major challenge in virtual screening is memory management when processing massive libraries. ScreenSAR addresses this through a **Chunk-based Processing Engine** in the prediction module. By reading and processing molecules in configurable batches (default: 5,000), and clearing transient descriptors from memory after each batch (garbage collection), the software maintains a constant, low memory footprint, allowing for the screening of millions of molecules on standard laptops without "Out of Memory" errors.

### **Results**
The implementation of ScreenSAR has resulted in a highly efficient tool that generates industry-ready outputs. Key findings from testing the pipeline include:
*   **Performance:** The chunking mechanism consistently processes libraries of over 1,000,000 compounds with stable RAM usage.
*   **Validation:** Automated ranking by MCC ensures that the most robust model is prioritized for screening. Statistical summaries, including ROC curves and confusion matrices, are automatically generated and formatted into professional PDF reports.
*   **User Adoption:** The multilingual support (English, Portuguese, Spanish, German, Chinese) and the intuitive dashboard allow researchers to navigate from raw data to a validated screening model in a single session.

### **Discussion**
The ScreenSAR user interface is designed as an interactive dashboard that guides the user through the data lifecycle. Unlike existing commercial suites like Pipeline Pilot or AutoQSAR, which can be prohibitively expensive and operate as "black boxes," ScreenSAR is open-source and provides full transparency of the underlying algorithms. While tools like KNIME offer high flexibility, they require a steep learning curve to build complex workflows; ScreenSAR provides these workflows pre-configured and ready for use. 

**Intended Uses and Benefits:** ScreenSAR is intended for medicinal chemists and researchers who need a robust QSAR pipeline without requiring deep coding skills. The primary benefit is the democratization of high-quality data curation and modeling, ensuring that even small research groups can perform large-scale virtual screening locally and privately.

**Future Development:** Planned features include the integration of deep learning architectures (such as Graph Neural Networks) and native cloud scaling to further accelerate the processing of ultra-large chemical universes (>100 million compounds).

### **Conclusions**
ScreenSAR represents a significant advancement in making high-throughput predictive modeling accessible to the global research community. By automating the most labor-intensive aspects of QSAR—data curation and memory-intensive screening—it allows scientists to focus on biological interpretation and strategic decision-making. The software's ability to handle massive datasets on standard hardware, combined with its focus on data integrity and scientific rigor, makes it a relevant and powerful tool for modern drug discovery.
