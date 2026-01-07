from curadoria import CuradoriaQSAR
import pandas as pd
import os

def create_dummy_data(filename):
    data = {
        'Molecule ChEMBL ID': ['CHEMBL1', 'CHEMBL2', 'CHEMBL1_dup_bad', 'CHEMBL3', 'CHEMBL4'],
        'Smiles': [
            'CCO', # Simple ethanol
            'CC.Cl', # Ethane with salt
            'CCO', # Duplicate of ethanol (to test consistency)
            'c1ccccc1', # Benzene
            'invalid_smiles' # Invalid
        ],
        'Standard Value': [100, 50, 10000, 20, 10],
        'Standard Units': ['nM', 'nM', 'nM', 'ug.mL-1', 'nM'],
        'Molecular Weight': [46.07, 30.07, 46.07, 78.11, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created dummy data at {filename}")

if __name__ == "__main__":
    input_file = 'dummy_data.csv'
    create_dummy_data(input_file)
    
    # Run the pipeline
    pipeline = CuradoriaQSAR(input_file)
    result = pipeline.executar_pipeline()
    
    if result is not None:
        print("\nResult DataFrame:")
        print(result.head())
        result.to_csv('curated_output.csv', index=False)
        print("\nSaved curated data to curated_output.csv")
    else:
        print("\nPipeline failed.")
