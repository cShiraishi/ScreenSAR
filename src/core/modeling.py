import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve, f1_score

class ModeladorQSAR:
    def __init__(self, df_input):
        self.df = df_input.copy()

    def gerar_dados(self, n_bits=1024, radius=2, descriptor_type="Morgan"):
        """Gera X (fingerprints) e y (outcome)"""
        fps = []
        outcomes = []
        valid_indices = []

        for idx, row in self.df.iterrows():
            smiles = row.get('SMILES_Clean')
            outcome = row.get('Outcome')
            
            if pd.isna(smiles) or pd.isna(outcome):
                continue
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if descriptor_type == "MACCS":
                        from rdkit.Chem import MACCSkeys
                        fp = MACCSkeys.GenMACCSKeys(mol)
                    elif descriptor_type == "RDKit":
                        fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=n_bits, nBitsPerHash=2)
                    else: # Default or Morgan
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                        
                    fps.append(np.array(fp))
                    outcomes.append(int(outcome))
                    valid_indices.append(idx)
            except:
                continue
        
        X = np.array(fps)
        y = np.array(outcomes)
        return X, y, valid_indices

    def treinar_avaliar(self, X, y, modelos_selecionados, test_size=0.2, random_state=42):
        """
        Treina e avalia os modelos selecionados.
        Retorna:
            - results (DataFrame): Métricas de performance.
            - trained_models (dict): Instâncias dos modelos treinados.
            - roc_data (dict): Dados para plotar curvas ROC (fpr, tpr, auc).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        results_list = []
        trained_models = {}
        roc_data = {}
        
        # Dicionário de construtores de modelo
        model_constructors = {
            "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=random_state),
            "SVM": lambda: SVC(probability=True, random_state=random_state), # probability=True needed for ROC
            "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=random_state),
            "KNN": lambda: KNeighborsClassifier(),
            "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=random_state)
        }
        
        for nome_modelo in modelos_selecionados:
            if nome_modelo in model_constructors:
                try:
                    clf = model_constructors[nome_modelo]()
                    clf.fit(X_train, y_train)
                    
                    # Previsões
                    y_pred = clf.predict(X_test)
                    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
                    
                    # Métricas
                    acc = accuracy_score(y_test, y_pred)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                    
                    results_list.append({
                        "Modelo": nome_modelo,
                        "Acurácia": acc,
                        "F1-Score": f1,
                        "MCC": mcc,
                        "Sensibilidade": sens,
                        "Especificidade": spec,
                        "AUC": auc,
                        "TP": tp,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn
                    })
                    
                    trained_models[nome_modelo] = clf
                    
                    if y_proba is not None:
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_data[nome_modelo] = {"fpr": fpr, "tpr": tpr, "auc": auc}
                        
                except Exception as e:
                    print(f"Erro ao treinar {nome_modelo}: {e}")
                    results_list.append({
                        "Modelo": nome_modelo,
                        "Acurácia": 0,
                        "F1-Score": 0,
                        "MCC": 0,
                        "Sensibilidade": 0,
                        "Especificidade": 0,
                        "AUC": 0,
                        "Erro": str(e)
                    })
                    
        results_df = pd.DataFrame(results_list)
        return results_df, trained_models, roc_data

    def calcular_modi(self, X, y):
        """
        Calcula o Índice de Modelabilidade (MODI).
        MODI = sum(I(Class_i == Class_NN_i)) / N
        Argumentos:
            X: Fingerprints
            y: Classes (0 e 1)
        Retorna:
            float: Valor MODI
        """
        from sklearn.neighbors import NearestNeighbors
        
        if len(y) < 2:
            return 0.0
            
        # Find nearest neighbor (k=2 because the closest is the point itself)
        nn = NearestNeighbors(n_neighbors=2, metric='jaccard') # Jaccard is common for bit vectors
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        
        matches = 0
        total = len(y)
        
        for i in range(total):
            # Index 0 is self, Index 1 is nearest neighbor
            idx_self = indices[i][0]
            idx_neighbor = indices[i][1]
            
            if y[idx_self] == y[idx_neighbor]:
                matches += 1
                
        modi = matches / total
        return modi
