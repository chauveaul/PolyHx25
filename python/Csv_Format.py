import csv

def replace_semicolon_with_comma_space(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile, delimiter=';')
        
        for row in reader:
            outfile.write(', '.join(row) + '\n')

if __name__ == "__main__":
    input_csv = r"C:\Users\lufai\Documents\new_list.csv"  # Replace with your input file
    output_csv = r"C:\Users\lufai\Documents\new_list2.csv"  # Replace with your desired output file
    replace_semicolon_with_comma_space(input_csv, output_csv)
    print(f"File '{output_csv}' has been created successfully.")



