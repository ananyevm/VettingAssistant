import pandas as pd
import json
import os
import numpy as np

def check_dominance_rule(csv_file, json_file):
    df = pd.read_csv(csv_file)
    violations = []

    for index, row in df.iterrows():
        numeric_values = [value for value in row if isinstance(value, (int, float, np.number))]
        row_sum = sum(numeric_values)

        if row_sum == 0 or pd.isna(row_sum):
            continue

        for col, value in row.items():
            if isinstance(value, (int, float, np.number)):
                dominance_ratio = value / row_sum
                if dominance_ratio > 0.8:
                    violations.append({
                        "row": index,
                        "column": col,
                        "value": value.item() if isinstance(value, (np.generic)) else value,
                        "row_sum": row_sum,
                        "dominance_ratio": dominance_ratio,
                        "position": [index, row.index.get_loc(col)]
                    })

    if not os.path.exists('json'):
        os.makedirs('json')

    with open(json_file, 'w') as f:
        json.dump({"violations": violations}, f)

    print(f"Total rows checked: {len(df)}")
    print(f"Total violations found: {len(violations)}")
    return json_file

check_dominance_rule('intermediate/supporting_cleaned.csv', 'json/dominance_violations.json')