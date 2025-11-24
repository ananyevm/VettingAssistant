import pandas as pd
import json
import os

def check_rule_of_10(csv_file, json_file):
    print(f"Reading file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame:\n{df}")
    print(f"Data types:\n{df.dtypes}")
    
    violations = []

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell_value = df.iat[row, col]
            column_name = df.columns[col]
            print(f"Row {row+1}, Col {column_name}: value={cell_value}, type={type(cell_value)}")
            
            if pd.isna(cell_value):
                print(f"  -> Skipping NaN")
                continue
                
            if isinstance(cell_value, (int, float)) and cell_value < 10:
                print(f"  -> Compliant (< 10)")
                continue
                
            if isinstance(cell_value, (int, float)) and cell_value >= 10:
                print(f"  -> VIOLATION (>= 10)")
                violations.append({
                    "row": row + 1,
                    "column": column_name,
                    "value": cell_value,
                    "position": [row + 1, col + 1]
                })

    print(f"Total violations found: {len(violations)}")
    return violations

if __name__ == "__main__":
    csv_path = 'intermediate/supporting_cleaned.csv'
    violations = check_rule_of_10(csv_path, 'debug.json')
    for v in violations:
        print(f"Violation: {v}")