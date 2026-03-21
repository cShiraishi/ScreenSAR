import pandas as pd
import io

data = b"""ChEMBL ID;Molecule Name;SMILES;Standard Type;Standard Value
CHEMBL1;Aspirin, pain;C(C(=O)O)C;IC50;10
CHEMBL2;Tylenol, pain;CC(=O)NC;IC50;20"""

f = io.BytesIO(data)
f.name = "DOWNLOAD-8dRaz_qDYU-J_i5rS_2Bc9QINR090J70prRwzmvqaiY_eq_-2.csv"
try:
    df = pd.read_csv(f, sep=None, engine='python')
    print("Columns:", list(df.columns))
except Exception as e:
    print("Error:", e)
