import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem import AllChem

class CuradoriaQSAR:
    def __init__(self, dados_entrada, corte_ativo_nm=100):
        self.corte = corte_ativo_nm
        self.salt_remover = SaltRemover.SaltRemover()

        if isinstance(dados_entrada, pd.DataFrame):
            self.df = dados_entrada.copy()
        elif isinstance(dados_entrada, str):
            # Determine if input is csv or excel based on extension
            if dados_entrada.endswith('.xlsx') or dados_entrada.endswith('.xls'):
                self.df = pd.read_excel(dados_entrada)
            else:
                self.df = pd.read_csv(dados_entrada)
        else:
            raise ValueError("Input must be a pandas DataFrame or a file path string.")
        
    def _limpar_quimica(self, smiles):
        """Fase 1: Normalização, Remoção de Sais e Tautômeros"""
        if pd.isna(smiles): return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            # 1. Remover Sais/Fragmentos menores
            # (Mantém apenas o maior fragmento orgânico)
            mol = self.salt_remover.StripMol(mol)
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags: return None # Handle case where StripMol leaves nothing
            mol = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
            
            # 2. Neutralizar e Normalizar (Tautômeros simples e Grupos funcionais)
            # RDKit faz boa parte disso automaticamente ao gerar o Canonical SMILES
            # Para normalização pesada, usaria MolStandardize (mais avançado)
            
            # 3. Gerar Canonical SMILES (Remove variações de desenho)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) 
            # isomericSmiles=False ignora estereoquímica se não for crucial, 
            # ajudando a encontrar duplicatas 2D.
        except:
            return None

    def _converter_unidades(self, row):
        """Fase 2: Conversão de Unidades"""
        try:
            val = float(row['Standard Value'])
            unit = str(row['Standard Units'])
            
            # Handle potential missing MW
            mw = row.get('Molecular Weight')
            if pd.isna(mw):
                 return None
            mw = float(mw)
            
            if unit == 'nM': return val
            if unit == 'ug.mL-1' and mw > 0: return (val * 1e6) / mw
            return None
        except:
            return None

    def executar_pipeline(self, calculate_pIC50=False):
        print("--- Iniciando Curadoria Profissional ---")
        
        # 1. Curadoria Química
        print("1. Padronizando Estruturas Químicas...")
        if 'Smiles' not in self.df.columns:
             print("Erro: Coluna 'Smiles' não encontrada.")
             return None

        self.df['SMILES_Clean'] = self.df['Smiles'].apply(self._limpar_quimica)
        self.df = self.df.dropna(subset=['SMILES_Clean'])
        
        # 2. Curadoria Biológica (Unidades)
        print("2. Convertendo Unidades...")
        # Check required columns for unit conversion
        required_cols = ['Standard Value', 'Standard Units', 'Molecular Weight']
        if not all(col in self.df.columns for col in required_cols):
             print(f"Erro: Colunas necessárias para conversão de unidades ausentes: {required_cols}")
             return None

        self.df['IC50_nM'] = self.df.apply(self._converter_unidades, axis=1)
        self.df = self.df.dropna(subset=['IC50_nM'])
        
        # 3. Análise de Duplicatas (Discordância)
        print("3. Analisando Duplicatas...")
        # Agrupa pelo SMILES Limpo (Canonical)
        grupo = self.df.groupby('SMILES_Clean')['IC50_nM']
        
        # Identificar duplicatas com alta variância (ex: desvio padrão alto)
        # Aqui simplificamos: removemos se a diferença max/min for > 10x (1 log)
        def verificar_discordancia(x):
            if len(x) > 1:
                ratio = x.max() / x.min()
                if ratio > 10: return 'Discordante'
            return 'Aceitavel'

        discordancia = grupo.apply(verificar_discordancia)
        smiles_ruins = discordancia[discordancia == 'Discordante'].index
        
        print(f"   -> Removidos {len(smiles_ruins)} compostos por discordância experimental.")
        self.df = self.df[~self.df['SMILES_Clean'].isin(smiles_ruins)]
        
        # Calcular média geométrica para os restantes
        agg_dict = {
            'IC50_nM': lambda x: np.exp(np.mean(np.log(x))),
        }
        if 'Molecule ChEMBL ID' in self.df.columns:
            agg_dict['Molecule ChEMBL ID'] = 'first' # Mantém um ID de referência
            
        df_final = self.df.groupby('SMILES_Clean').agg(agg_dict).reset_index()
        
        # Optional: Calculate pIC50
        if calculate_pIC50:
             # pIC50 = -log10(Molar). IC50_nM is nanoMolar (10^-9).
             # pIC50 = -log10(IC50_nM * 10^-9) = 9 - log10(IC50_nM)
             # Handling log(0) or negative (shouldn't exist but safe to check)
             df_final['pIC50'] = df_final['IC50_nM'].apply(
                 lambda x: 9 - np.log10(x) if x > 0 else None
             )
        
        # 4. Definição de Classes
        df_final['Outcome'] = df_final['IC50_nM'].apply(lambda x: 1 if x <= self.corte else 0)
        
        
        print(f"--- Finalizado. Total de Compostos Únicos: {len(df_final)} ---")
        return df_final

    def gerar_fingerprints(self, df_input, n_bits=1024, radius=2):
        """Gera Morgan Fingerprints para uma coluna de SMILES"""
        fps = []
        valid_indices = []
        
        for idx, row in df_input.iterrows():
            smiles = row['SMILES_Clean']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    fps.append(np.array(fp))
                    valid_indices.append(idx)
            except:
                continue
                
        return np.array(fps), valid_indices
