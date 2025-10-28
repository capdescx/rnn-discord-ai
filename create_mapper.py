import json
import csv
import re

with open('cleaned_vocab.json', 'r', encoding='utf-8') as vocab_file:
    vocab = json.load(vocab_file)

pattern = r'(\w+|[^\w\s])'

processed_data_2 = []
with open('dataset.csv', 'r', encoding='utf-8') as dataset_file:
    csv_reader = csv.DictReader(dataset_file)
    for row in csv_reader:
        input_text = row['Input']
        output_text = row['Output']
        
        input_tokens = [str(vocab.get(token, 'UNK')) for token in re.findall(pattern, input_text)]
        output_tokens = [str(vocab.get(token, 'UNK')) for token in re.findall(pattern, output_text)]
        
        processed_data_2.append({
            'Input': ' '.join(input_tokens),
            'Output': ' '.join(output_tokens)
        })

with open('processed_data.csv', 'w', newline='', encoding='utf-8') as processed_file:
    fieldnames = ['Input', 'Output']
    csv_writer = csv.DictWriter(processed_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(processed_data_2)

print("Tokenization and processing completed.")
