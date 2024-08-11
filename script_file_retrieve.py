import os
import shutil

def copy_files_with_ids(source_dir, target_dir, id_list):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        # Check if any ID from the list is in the filename
        if any(id in filename for id in id_list):
            # Construct full file paths
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            # Copy the file to the target directory
            shutil.copy2(source_path, target_path)
            print(f"Copied: {filename}")

# Example usage
source_directory = "data/strict-llama3-strict-gold"
target_directory = "data/resolve/set2/llama3"
worksheet_list = [
    "10002221-DS-11", "10004401-DS-22", "10004401-DS-29", "10094971-DS-3",
    "10018052-DS-17", "10024331-DS-28", "10024331-DS-29", "10024331-DS-31",
    "10035631-DS-13", "10094971-DS-5", "10041127-DS-17", "10041836-DS-20",
    "10047172-DS-15", "10047172-DS-16", "10052992-DS-17", "10054464-DS-19",
    "10054464-DS-20", "10056223-DS-4", "10059192-DS-10", "10060764-DS-8",
    "10060764-DS-9", "10070201-DS-19", "10070594-DS-14", "10070594-DS-16",
    "10073847-DS-30", "10074556-DS-22", "10074858-DS-16", "10076342-DS-20",
    "10076617-DS-11", "10076958-DS-13", "10078297-DS-5", "10078933-DS-9",
    "10079616-DS-8", "10079616-DS-9", "10084586-DS-19", "10085005-DS-5",
    "10085725-DS-12", "10089085-DS-18", "10090755-DS-7", "10090755-DS-8",
    "10091141-DS-20", "10095417-DS-19", "10091385-DS-16", "10091385-DS-17",
    "10091873-DS-22", "10093120-DS-18", "10097898-DS-11", "10098672-DS-3",
    "10036086-DS-25", "10098875-DS-12"
] # Replace with your actual ID list

copy_files_with_ids(source_directory, target_directory, worksheet_list)