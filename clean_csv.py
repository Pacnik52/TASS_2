import csv

# Input and output file paths
input_file = 'school_thresholds_otouczelnie_raw.csv'
output_file = 'school_thresholds_otouczelnie_clean.csv'

# Process the CSV
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        if len(row) >= 5:  # Ensure we have at least the expected columns
            # Clean the address field (column 3, 0-indexed)
            address = row[3]
            # Replace newlines and multiple spaces with single space
            address = ' '.join(address.split())
            row[3] = address
        
        # Write the cleaned row
        writer.writerow(row)

print(f"Cleaned CSV saved to {output_file}")