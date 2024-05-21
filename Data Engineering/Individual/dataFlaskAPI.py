import os
import json
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/Shahron/prisonstatisticsapi', methods=['GET'])
def get_csv_json():
    base_path = os.getcwd()
    folder_path = "Cleaned"  # Update this to your folder path
    path = os.path.join(base_path,folder_path)
    print(path)
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')][:5]  # Get the first 5 CSV files
    
    if len(csv_files) < 5:
        return jsonify({"error": "Less than 5 CSV files found"}), 400

    csv_data = {}

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        file = file.replace(".csv", "")
        csv_data[file] = df.to_dict(orient='records')
    
    return jsonify(csv_data)

@app.route('/Shahron/metadataapi', methods=['GET'])
def get_json_file():
    folder_path = os.getcwd()  # Update this to your JSON folder path in WSL
    file_path = os.path.join(folder_path, "metaData.json")

    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return jsonify(json_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000)
    

