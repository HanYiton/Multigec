import argparse
import os

def load_domain_mapping(domain_file):
    """
    Loads the domain mapping from a file, where each line contains a text ID and its corresponding domain.
    Returns a dictionary mapping text IDs to domains.
    """
    domain_mapping = {}
    with open(domain_file, 'r', encoding='utf-8') as f:
        for line in f:
            text_id, domain = line.strip().split('\t')
            domain_mapping[text_id] = domain
    return domain_mapping

def split_by_domain(input_file, domain_file, output_dir):
    """
    Splits the input file into multiple output files based on domain mappings.
    """
    # Load the domain mapping
    domain_mapping = load_domain_mapping(domain_file)

    # Extract base name of input file (without directory and extension)
    base_filename = os.path.basename(input_file)
    
    # Open file handles for each domain with the modified filenames
    domain_files = {domain: open(f"{output_dir}/{domain}_{base_filename}", 'w', encoding='utf-8') 
                    for domain in set(domain_mapping.values())}

    current_text_id = None
    current_domain = None
    text_buffer = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.startswith("### essay_id ="):
                # Write the previous text to its respective domain file, if any
                if current_text_id is not None and text_buffer:
                    # Join text buffer with exactly one blank line between texts (no trailing newlines)
                    text_to_write = "".join(text_buffer).rstrip("\n") + "\n\n"
                    domain_files[current_domain].write(text_to_write)
                
                # Extract text ID and determine the domain
                current_text_id = line.split("### essay_id = ")[1].strip()
                current_domain = domain_mapping.get(current_text_id, None)

                # Start a new text buffer for this essay
                text_buffer = [line] if current_domain else []
            elif current_domain:
                # Append to the current essay's buffer if the domain is known
                text_buffer.append(line)

        # Write the last text to its respective domain file, if any
        if current_text_id is not None and text_buffer:
            text_to_write = "".join(text_buffer).rstrip("\n") + "\n"
            domain_files[current_domain].write(text_to_write)

    # Close all the output files
    for file in domain_files.values():
        file.close()

def main():
    parser = argparse.ArgumentParser(description='Split essays into multiple files based on their domain.')
    parser.add_argument('input_file', help='Path to the input text file containing essays.')
    parser.add_argument('domain_file', help='Path to the file mapping text IDs to domains.')
    parser.add_argument('output_dir', help='Directory to save the output files.')

    args = parser.parse_args()
    split_by_domain(args.input_file, args.domain_file, args.output_dir)

if __name__ == '__main__':
    main()
