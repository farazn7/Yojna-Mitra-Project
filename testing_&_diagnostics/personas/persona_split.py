import json
import os

def split_personas():
    input_file = 'personas.json'
    
    # Make sure the file exists before trying to open it
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}. Make sure you are running this from the project root.")
        return

    # Read the master list of 10 personas
    with open(input_file, 'r', encoding='utf-8') as f:
        all_personas = json.load(f)

    # We want items from index 2 up to 9 (which are personas 3 through 10)
    for i in range(0, 10):
        # Extract the single persona object
        single_persona = all_personas[i]
        
        # Format the file name exactly like the ones in your image
        output_filename = f'person{i+1}.json'
        
        # Write it to its own file
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            json.dump(single_persona, out_f, indent=2, ensure_ascii=False)
            
        print(f"✅ Created {output_filename}")

if __name__ == "__main__":
    split_personas()
    print("Done splitting!")