import os
import json
from tqdm import tqdm
# Initialize the directory containing JSON files and the output text file
base_path = '/home/bishoymoussas/Workspace/MOT/R_MTT5/R_MTT5_subscenes/subscene_2/'
json_directory = base_path  # Replace with your directory
output_txt_file = base_path+"gt.txt"

# Open the output text file
with open(output_txt_file, "w") as out_file:
    # Loop through each JSON file
    for json_file in tqdm(os.listdir(json_directory)):
        if json_file.endswith(".json"):
            frame_id = os.path.splitext(json_file)[0]  # Extract frame_id from file name
            json_path = os.path.join(json_directory, json_file)
            
            # Read JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Parse and write to output file
            for shape in data['shapes']:
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                label = int(shape['label'])  # Assuming label is an integer
                
                # Write to output file
                out_file.write(f"{frame_id},{label},{x1},{y1},{x2},{y2}\n")
