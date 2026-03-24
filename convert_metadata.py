import os
import json
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'arm_angle')
CSV_PATH = '/tmp/arm_angle_docs/spreadsheet.csv'
JSON_PATH = os.path.join(DATA_DIR, 'metadata.json')

def convert():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        print(f"Cannot find {CSV_PATH}. Make sure it is downloaded.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    metadata = {}
    for _, row in df.iterrows():
        subject = str(row['Subject']).strip()
        if subject == 'nan' or not subject:
            continue
            
        comment = str(row.get('comment', ''))
        exclude = 'Exclude' in comment
        
        try:
            vivid = int(row.get('Vivid (1 to 5 (5 = very vivid))', 0))
        except:
            vivid = 0
            
        angles = []
        for col in ['1', '2', '3', '4', '5']:
            try:
                angles.append(float(row[col]))
            except:
                angles.append(None)
                
        metadata[subject] = {
            'file': str(row.get('File', '')),
            'angles': angles,
            'vividness': vivid,
            'exclude': exclude,
            'comment': comment
        }
        
    with open(JSON_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata properly to {JSON_PATH}")

if __name__ == '__main__':
    convert()
