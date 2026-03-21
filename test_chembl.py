import pandas as pd
import io

class MockUploadedFile(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

file_content = b"ChEMBL ID;Molecule Name;SMILES;Standard Type;Standard Value\nCHEMBL1;;A;IC50;10\nCHEMBL2;;B;IC50;20"
long_name = "DOWNLOAD-8dRaz_qDYU-J_i5rS_2Bc9QINR090J70prRwzmvqaiY_eq_-2.csv"
file_obj = MockUploadedFile(file_content, long_name)

try:
    df = pd.read_csv(file_obj, sep=None, engine='python')
    print("Success:", df.shape)
except Exception as e:
    print("Error:", repr(e))
