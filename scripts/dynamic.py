import os
import csv


# folder_path = '12Lab_NMTV_Video_To_Text'
folder_path = '12Lab_MTV_Video_To_Text'
output_csv = 'dynamic.csv'

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    if csv_file.tell() == 0:
        csv_writer.writerow(['Video ID', 'Video to Text Description', 'Label'])

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        filename_without_ext = os.path.splitext(txt_file)[0]
        # csv_writer.writerow([filename_without_ext, content, "NMTV"])
        csv_writer.writerow([filename_without_ext, content, "MTV"])

print(f"Data has been written to {output_csv} successfully.")
