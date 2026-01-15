import pandas as pd
import requests
import os
from urllib.parse import urlparse, parse_qs

def get_video_id(youtube_url):
    """
    Extract video ID from a YouTube URL
    """
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname == 'youtu.be':
        # Handle shortened URL format: https://youtu.be/ID
        return parsed_url.path[1:]
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        # Handle full URL format: https://www.youtube.com/watch?v=ID
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    return None

def download_thumbnail(video_id, save_path):
    """
    Download the YouTube thumbnail based on the video ID
    """
    url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded thumbnail for video ID {video_id}")
        return True
    else:
        print(f"Failed to download thumbnail for video ID {video_id}")
        return False

def main(csv_file, output_folder, id_column='url', output_csv='no-thumb.csv'):
    """
    Main function to download thumbnails and log failed URLs
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List to store the URLs with failed thumbnail retrieval
    failed_urls = []
    
    # Loop through the URLs and download thumbnails
    for url in df[id_column]:
        video_id = get_video_id(url)
        if video_id:
            save_path = os.path.join(output_folder, f"{video_id}.jpg")
            if not download_thumbnail(video_id, save_path):
                # Add the video URL to the list of failed thumbnails
                failed_urls.append(url)
        else:
            print(f"Invalid YouTube URL: {url}")
            failed_urls.append(url)
    
    # If there are failed URLs, write them to a CSV file (outside the output folder)
    if failed_urls:
        failed_df = pd.DataFrame(failed_urls, columns=[id_column])
        failed_df.to_csv(output_csv, index=False)  # Save the CSV in the current working directory
        print(f"Failed thumbnail URLs saved to {output_csv}")
    else:
        print("All thumbnails downloaded successfully.")

if __name__ == "__main__":
    """For ease in evaluation/testing store the MTV and NMTV urls in separate csvs and name the output folder as MTV_Thumbnails or NMTV_Thumbnails"""
    csv_file = "/path/to/your/csv/"
    id_column = "url"  # The column that contains the YouTube video URLs or IDs
    # output_folder = "NMTV_Thumbnails"
    output_folder = "MTV_Thumbnails"
    main(csv_file, output_folder, id_column)
