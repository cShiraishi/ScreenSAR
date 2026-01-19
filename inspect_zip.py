import zipfile
import sys

file_path = 'ptp11bb_0.csv'

print(f"Inspecting ZIP contents of {file_path}...")

try:
    with zipfile.ZipFile(file_path, 'r') as z:
        print("Files inside ZIP:")
        for name in z.namelist():
            print(f" - {name}")
except zipfile.BadZipFile:
    print("Not a valid ZIP file.")
except Exception as e:
    print(f"Error: {e}")
