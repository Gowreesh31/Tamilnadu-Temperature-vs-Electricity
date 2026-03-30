import pandas as pd
import os

# Set the path to your raw data folder
data_dir = 'data/raw/'

print(f"{'File Name':<40} | {'Rows':<10} | {'Columns':<10}")
print("-" * 65)

# Walk through all folders and files
for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        
        try:
            # Check if it's a CSV
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"{file:<40} | {df.shape[0]:<10} | {df.shape[1]:<10}")
            
            # Check if it's an Excel file
            elif file.endswith('.xlsx') or file.endswith('.xls'):
                df = pd.read_excel(file_path)
                print(f"{file:<40} | {df.shape[0]:<10} | {df.shape[1]:<10}")
                
        except Exception as e:
            print(f"{file:<40} | Error reading file")
