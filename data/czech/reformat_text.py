import argparse

def reformat_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        current_text_id = None
        text_buffer = []

        for line in infile:
            if not line.strip():
                continue  # Skip empty lines if any

            text_id, paragraph = line.split('\t', 1)

            if text_id != current_text_id:
                # If a new text ID is encountered, write the previous text to the output file
                if current_text_id is not None:
                    outfile.write(f"### essay_id = {current_text_id}\n")
                    outfile.write("".join(text_buffer))
                    outfile.write("\n")  # Separate texts by a single new line

                # Reset buffer and update current_text_id
                current_text_id = text_id
                text_buffer = [paragraph]
            else:
                # If the text ID is the same, append the paragraph to the buffer
                text_buffer.append(paragraph)

        # Write the last buffered text if any
        if text_buffer:
            outfile.write(f"### essay_id = {current_text_id}\n")
            outfile.write("".join(text_buffer))

def main():
    parser = argparse.ArgumentParser(description='Reformat texts by consolidating paragraphs and labeling text IDs only once.')
    parser.add_argument('input_file', help='Path to the input text file.')
    parser.add_argument('output_file', help='Path to the output text file.')

    args = parser.parse_args()
    reformat_text(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
