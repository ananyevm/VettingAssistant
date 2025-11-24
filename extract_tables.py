import pandas as pd
import os
from pathlib import Path

def extract_tables_from_excel(file_path, output_dir):
    """Extract all sheets from Excel file and save as CSV files"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(file_path)
        file_name = Path(file_path).stem
        
        print(f"Processing {file_name}...")
        print(f"Found sheets: {excel_file.sheet_names}")
        
        for sheet_name in excel_file.sheet_names:
            # Read each sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Create output filename
            output_filename = f"{file_name}_{sheet_name}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as CSV
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_filename} ({len(df)} rows, {len(df.columns)} columns)")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Create intermediate directory if it doesn't exist
    intermediate_dir = "intermediate"
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Extract tables from output files
    test_data_dir = "test_data/test_case_1"
    
    output_file = os.path.join(test_data_dir, "output.xlsx")
    supporting_file = os.path.join(test_data_dir, "supporting.xlsx")
    
    print("Extracting tables from Excel files...")
    extract_tables_from_excel(output_file, intermediate_dir)
    extract_tables_from_excel(supporting_file, intermediate_dir)
    
    print(f"\nAll tables extracted to '{intermediate_dir}' folder")

if __name__ == "__main__":
    main()