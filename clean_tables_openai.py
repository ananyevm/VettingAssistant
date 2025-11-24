import pandas as pd
import os
import json
from pathlib import Path
from openai import OpenAI

def extract_raw_data_from_excel(file_path):
    """Extract raw data from Excel file as text representation"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(file_path)
        all_data = {}
        
        for sheet_name in excel_file.sheet_names:
            # Read sheet without any processing
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            # Convert to string representation
            all_data[sheet_name] = df.to_string(index=False, header=False)
            
        return all_data
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def clean_table_with_openai(raw_data, filename, client):
    """Use OpenAI to clean and structure table data"""
    
    prompt = f"""
You are a data cleaning expert. I have raw table data extracted from an Excel file that may contain:
- Merged cells
- Headers in wrong positions
- Empty rows/columns
- Inconsistent formatting
- Extraneous information

Please analyze this raw data and extract clean, well-structured tabular data in CSV format.

Raw data from file '{filename}':
{raw_data}

Requirements:
1. Identify the actual table structure and headers
2. Remove any extraneous information
3. Ensure consistent data types in each column
4. Handle any merged cells or formatting issues
5. Return ONLY the clean CSV data (no explanations)
6. Use proper column headers
7. Remove any completely empty rows or columns

Return the cleaned data as a proper CSV format.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data cleaning expert. Return only clean CSV data with no additional text or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None

def save_cleaned_csv(cleaned_data, output_path):
    """Save cleaned data as CSV file"""
    try:
        # Remove markdown code block markers if present
        if cleaned_data.startswith('```csv'):
            cleaned_data = cleaned_data[6:]  # Remove ```csv
        if cleaned_data.startswith('```'):
            cleaned_data = cleaned_data[3:]  # Remove ```
        if cleaned_data.endswith('```'):
            cleaned_data = cleaned_data[:-3]  # Remove trailing ```
        
        # Clean up the data
        cleaned_data = cleaned_data.strip()
        
        # Write the cleaned data
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_data)
        return True
    except Exception as e:
        print(f"Error saving CSV to {output_path}: {str(e)}")
        return False

def main():
    # Initialize OpenAI client
    client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    
    # Create intermediate directory if it doesn't exist
    intermediate_dir = "intermediate"
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Process each Excel file
    test_data_dir = "test_data/test_case_1"
    files_to_process = [
        ("output.xlsx", "output"),
        ("supporting.xlsx", "supporting")
    ]
    
    for excel_filename, base_name in files_to_process:
        excel_path = os.path.join(test_data_dir, excel_filename)
        print(f"\nProcessing {excel_filename}...")
        
        # Extract raw data
        raw_data_dict = extract_raw_data_from_excel(excel_path)
        
        if raw_data_dict:
            for sheet_name, raw_data in raw_data_dict.items():
                print(f"Cleaning sheet: {sheet_name}")
                
                # Clean data with OpenAI
                cleaned_data = clean_table_with_openai(raw_data, f"{excel_filename}_{sheet_name}", client)
                
                if cleaned_data:
                    # Save cleaned CSV
                    output_filename = f"{base_name}_{sheet_name}_cleaned.csv"
                    output_path = os.path.join(intermediate_dir, output_filename)
                    
                    if save_cleaned_csv(cleaned_data, output_path):
                        print(f"✓ Saved cleaned data: {output_filename}")
                        
                        # Validate the cleaned CSV
                        try:
                            df_check = pd.read_csv(output_path)
                            print(f"  - Shape: {df_check.shape}")
                            print(f"  - Columns: {list(df_check.columns)}")
                        except Exception as e:
                            print(f"  - Warning: Could not validate CSV: {e}")
                    else:
                        print(f"✗ Failed to save: {output_filename}")
                else:
                    print(f"✗ Failed to clean data for {sheet_name}")
    
    print(f"\nProcessing complete. Check the '{intermediate_dir}' folder for cleaned CSV files.")

if __name__ == "__main__":
    main()