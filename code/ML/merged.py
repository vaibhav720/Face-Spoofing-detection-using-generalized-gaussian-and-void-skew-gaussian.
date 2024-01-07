import pandas as pd
import os

def merge_excel_files(folder_path, output_file):
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    merged_data = pd.DataFrame()
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, df], ignore_index=True)
    merged_data.to_csv(output_file, index=False)
    print(f"Completed")

folder_path = '/content/excel_file/'  
output_file = '/content/merged_data.csv'  

merge_excel_files(folder_path, output_file)
