import sys

# Function to create a mapping of IDs to indices
def create_id_mapping(file_path):
    id_set = set()
    # Read the file and collect all unique IDs
    with open(file_path, 'r') as file:
        for line in file:
            ids = line.split()
            id_set.update(ids)
    # Create a mapping from ID to index
    return {id: index for index, id in enumerate(sorted(id_set))}

# Main program
def main(input_file_path, output_file_path):
    # Create the ID to index mapping
    id_mapping = create_id_mapping(input_file_path)

    # Open the input file to read and the output file to write
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Split the line into IDs and map them to indices
            ids = line.split()
            mapped_ids = [str(id_mapping[id]) for id in ids]
            # Join the mapped indices with '1 +' and write to the output file
            new_line = "+ " + ' '.join(mapped_ids) + "\n"
            outfile.write(new_line)

    print(f"The lines have been modified with indices and written to '{output_file_path}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
    else:
        input_file, output_file = sys.argv[1], sys.argv[2]
        main(input_file, output_file)
