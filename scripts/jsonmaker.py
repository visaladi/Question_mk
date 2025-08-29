import os
import json
from glob import glob

# Input folder
input_folder = r"C:\Users\visal Adikari\OneDrive\Desktop\uni sem 7\fyp\project\Question_mk\outputs"

# Output folder + file
output_folder = r"C:\Users\visal Adikari\OneDrive\Desktop\uni sem 7\fyp\project\Question_mk\scripts\data"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
output_file = os.path.join(output_folder, "merged_output.json")

# Collect all JSON files
json_files = glob(os.path.join(input_folder, "*.json"))

merged_data = []

for file in json_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # If the file is a list, extend; if dict, append
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

# Write merged file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print(f"✅ Merged {len(json_files)} JSON files into {output_file}")
