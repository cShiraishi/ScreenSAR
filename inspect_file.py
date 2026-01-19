import pandas as pd
import os

file_path = 'ptp11bb_0.csv'

print(f"Inspecting {file_path}...")

# 1. Check Magic Bytes
with open(file_path, 'rb') as f:
    header = f.read(4)
    print(f"Magic bytes: {header}")
    if header.startswith(b'PK'):
        print("-> Detected ZIP/XLSX signature")
    elif header.startswith(b'\xd0\xcf'):
        print("-> Detected XLS signature")
    else:
        print("-> Likely text/CSV")

# 2. Try identifying separator if text
if not (header.startswith(b'PK') or header.startswith(b'\xd0\xcf')):
    try:
        with open(file_path, 'r', encoding='latin1') as f:
            first_line = f.readline()
            print(f"First line (latin1): {first_line.strip()}")
            if ';' in first_line: print("-> Detected semicolon separator")
            elif ',' in first_line: print("-> Detected comma separator")
            elif '\t' in first_line: print("-> Detected tab separator")
            
            # Count fields
            import csv
            f.seek(0)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(f.read(1024))
                print(f"-> Sniffer detected delimiter: '{dialect.delimiter}'")
            except Exception as e:
                print(f"-> Sniffer failed: {e}")
    except Exception as e:
        print(f"Error reading as text: {e}")
