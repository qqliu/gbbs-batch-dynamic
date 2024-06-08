import random

def shuffle_and_transform_lines(input_file, output_file):
    """
    Reads lines from an input file, removes leading "+", shuffles them, and
    writes them with a prepended "-" to an output file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the desired output text file.
    """

    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Remove leading "+" from each line (working on the copy)
    lines = [line.lstrip('+').strip() for line in lines if line.strip()]

    # Shuffle the lines (shuffling the modified copy)
    random.shuffle(lines)

    # Prepend "-" and write to output file (writing the modified lines)
    with open(output_file, 'w') as outfile:
        for line in lines:             # Then write the shuffled, modified lines
            outfile.write(f"- {line}\n")

# Get input and output file names from the user
input_filename = input("Enter the input file name: ")
output_filename = input("Enter the output file name: ")

# Call the function to process the files
shuffle_and_transform_lines(input_filename, output_filename)

print(f"Processing complete. Results written to {output_filename}")

