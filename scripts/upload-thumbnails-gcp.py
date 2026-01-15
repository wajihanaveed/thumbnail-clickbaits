import csv
import os
from pytubefix import YouTube
from pytubefix.exceptions import RegexMatchError, VideoUnavailable
from google.cloud import storage

# Initialize Google Cloud Storage client
storage_client = storage.Client()

def process_thumbnail(url):
    try:
        # Initialize YouTube object to get the video ID
        yt = YouTube(url)
        video_id = yt.video_id
        
        # Check if the thumbnail exists locally
        # thumbnail_file = f'NMTV_Thumbnails/{video_id}.jpg'
        thumbnail_file = f'MTV_Thumbnails/{video_id}.jpg'
        if os.path.exists(thumbnail_file):
            # Upload the thumbnail to the Google Cloud Storage bucket
            upload_to_gcs(thumbnail_file, '<BUCKET-NAME>', f'{video_id}.jpg')
            print(f"Thumbnail {thumbnail_file} uploaded successfully!")
        else:
            print(f"Thumbnail {thumbnail_file} not found.")
            
    except RegexMatchError as e:
        print(f"Error: {e}. The URL provided is not valid.")
    except VideoUnavailable as e:
        print(f"Error: {e}. The video is unavailable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Uploads a file to the specified Google Cloud Storage bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the thumbnail
        blob.upload_from_filename(local_file_path)
        
        print(f"Upload completed! {local_file_path} uploaded to {bucket_name}/{destination_blob_name}.")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

def process_csv(file_path):
    """Reads URLs from a CSV file and processes each one to upload thumbnails."""
    try:
        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                url = row['url']
                print(f"\nProcessing URL: {url}")
                process_thumbnail(url)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

if __name__ == "__main__":
    # Specify the path to the CSV file containing the video URLs
    csv_file_path = "/path/to/your/csv/"
    process_csv(csv_file_path)
