import os
import glob

worksheet_list = [
    "10000935-DS-21",
    "10000980-DS-23",
    "10001401-DS-20",
    "10054464-DS-17",
    "10002221-DS-12",
    "10003299-DS-10",
    "10056223-DS-14",
    "10004401-DS-26",
    "10056612-DS-8",
    "10006029-DS-16",
    "10006431-DS-24",
    "10006580-DS-21",
    "10006820-DS-18",
    "10007795-DS-13",
    "10008628-DS-3",
    "10010440-DS-5",
    "10011938-DS-16",
    "10012292-DS-9",
    "10014354-DS-23",
    "10016142-DS-19",
    "10017285-DS-3",
    "10018052-DS-16",
    "10057126-DS-7",
    "10020306-DS-9",
    "10021312-DS-20",
    "10021493-DS-18",
    "10022373-DS-5",
    "10057731-DS-7",
    "10059192-DS-11",
    "10024331-DS-30",
    "10060733-DS-12",
    "10025862-DS-12",
    "10027957-DS-18",
    "10032176-DS-14",
    "10034049-DS-20",
    "10035631-DS-12",
    "10060764-DS-6",
    "10061124-DS-12",
    "10041127-DS-16",
    "10062597-DS-7",
    "10041408-DS-18",
    "10062981-DS-5",
    "10041836-DS-21",
    "10043750-DS-6",
    "10067059-DS-15",
    "10067195-DS-13",
    "10047172-DS-17",
    "10052938-DS-2",
    "10052992-DS-11",
    "10052992-DS-16"
]

# Specify the directory path
directory_path = 'data/medical_notes_ids'

# Initialize an empty list to store filenames
file_list = []

# Iterate over all files in the directory
for filepath in glob.glob(os.path.join(directory_path, '*')):
    filename = os.path.basename(filepath)
    # Remove the 'oncology-report-' prefix and '.txt' suffix
    modified_filename = filename.replace('oncology-report-', '').replace('.txt', '')
    file_list.append(modified_filename)

# # Print the generated list
# print(file_list)

# Convert the lists to sets and find the difference
missing_in_folder = set(worksheet_list) - set(file_list)

# Print the missing elements
if missing_in_folder:
    print("Files missing in the folder:", ", ".join(missing_in_folder))
else:
    print("No files are missing in the folder.")

