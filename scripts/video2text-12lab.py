import os
import csv
from google.cloud import storage
from twelvelabs import TwelveLabs
from twelvelabs.models.task import Task
from urllib.parse import urlparse, parse_qs

API_KEY = "<api-key>"
client = TwelveLabs(api_key=API_KEY)

# Ensure the folder exists
# save_folder = "12Lab_NMTV_Video_To_Text"
save_folder = "12Lab_MTV_Video_To_Text"
os.makedirs(save_folder, exist_ok=True)

# Function to extract YouTube video ID from URL
def extract_video_id(youtube_url):
    url_data = urlparse(youtube_url)
    query = parse_qs(url_data.query)
    video_id = query["v"][0] if "v" in query else None
    return video_id

# Function to read YouTube URLs from a CSV and extract video IDs
def read_video_ids_from_csv(csv_file):
    video_ids = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if there's one
        for row in reader:
            youtube_url = row[0]
            video_id = extract_video_id(youtube_url)
            if video_id:
                video_ids.append(video_id)
    return video_ids

# Function to save the response to a text file in the specified folder
def save_response_to_file(video_id, response_text, folder):
    filename = os.path.join(folder, f"{video_id}.txt")
    with open(filename, 'w') as file:
        file.write(response_text)
    print(f"Response saved to {filename}")

# Function to get all videos in the specified index
def get_all_videos(index_id):
    videos = []
    page = 1

    while True:
        # Fetch a page of videos
        page_videos = client.index.video.list(index_id=index_id, page=page)
        
        if not page_videos:
            break  # No more videos to fetch
        
        videos.extend(page_videos)
        page += 1

    return videos

# Read video IDs from the CSV file
csv_file = "/path/to/your/csv/"  # Your CSV file containing YouTube URLs
video_ids_from_csv = read_video_ids_from_csv(csv_file)

# List all videos in the specified index
videos = get_all_videos("<index-id>")
print(f"Total videos fetched: {len(videos)}")

# Create a mapping of video filenames from Twelve Labs index for easy lookup
filename_to_video = {video.metadata.filename.replace(".mp4", ""): video for video in videos}

# Loop through video IDs from the CSV and search them in the Twelve Labs videos
for video_id in video_ids_from_csv:
    if video_id in filename_to_video:
        video = filename_to_video[video_id]
        try:
            print(f"Generating text for {video.metadata.filename}")
            res = client.generate.text(video_id=video.id, prompt="Watch the video and provide a detailed description. Break down the content scene by scene, focusing on key actions, visuals, and emotions.")
            
            if res and hasattr(res, 'data'):
                save_response_to_file(video_id, res.data, save_folder)
            else:
                print(f"Failed to retrieve data for video: {video.metadata.filename}")

        except Exception as e:
            print(f"Error generating text for {video.metadata.filename}: {e}")
    else:
        print(f"Video ID {video_id} not found in Twelve Labs index.")
