import csv

# Input and output file paths
input_file = r"C:\Users\lufai\Downloads\archive\wildfire_list.csv"  # Replace with your actual file
output_file = r"C:\Users\lufai\Downloads\archive\wildfire_list2.csv"  # Replace with desired output file

# Open the input file and replace ';' with ','
with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:
    
    reader = csv.reader(infile, delimiter=";")
    writer = csv.writer(outfile, delimiter=",")

    for row in reader:
        writer.writerow(row)

print("Conversion completed. Check", output_file)
