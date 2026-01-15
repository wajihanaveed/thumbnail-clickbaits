import os
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage

# Initialize Vertex AI and Google Cloud Storage

PROJECT_ID = "your-project-id"
vertexai.init(project=PROJECT_ID, location="us-east1")
storage_client = storage.Client()

# Load the CSV file
csv_file = "/path/to/your/csv/"
df = pd.read_csv(csv_file)

# Ensure the output directory exists
# output_dir = 'Gemini_NMTV_Video_To_Text'
output_dir = 'Gemini_MTV_Video_To_Text'
os.makedirs(output_dir, exist_ok=True)

# Unavailable videos file in CWD
unavailable_file = os.path.join(os.getcwd(), 'gemini_video_to_text_unavailable.txt')
issue_file = os.path.join(os.getcwd(), 'gemini_video_to_text_issue.txt')

# Ensure unavailable_file exists or create it in CWD
if not os.path.exists(unavailable_file):
    with open(unavailable_file, 'w') as f:
        pass  # Create the file if it doesn't exist

# Ensure issue_file exists or create it in CWD
if not os.path.exists(issue_file):
    with open(issue_file, 'w') as f:
        pass  # Create the file if it doesn't exist

# Initialize the model
vision_model = GenerativeModel("gemini-1.5-flash-001")

# Function to extract the video ID from a YouTube URL
def extract_video_id(url):
    if 'youtube.com/watch?v=' in url:
        return url.split('youtube.com/watch?v=')[-1].split('&')[0]
    else:
        return None

def check_video_exists(bucket_name, video_id):
    bucket = storage_client.bucket(bucket_name)
    # Update the path to check for the video in the "videos" folder
    blob = bucket.blob(f"videos/{video_id}.mp4")
    return blob.exists()

# Iterate over the URLs in the CSV
for url in df['url']:
    video_id = extract_video_id(url)
    if video_id:
        if check_video_exists('<bucket-name>', video_id):
            try:
                # Generate the content
                response = vision_model.generate_content(
                    [
                        Part.from_uri(f"gs://<bucket-name>/videos/{video_id}.mp4", mime_type="video/mp4"),
                        "Watch the video and provide a detailed description. Break down the content scene by scene, focusing on key actions, visuals, and emotions.",
                    ]
                )

                # Get the response text
                response_text = response.text

                # Save the response to a text file
                output_file = os.path.join(output_dir, f"{video_id}.txt")
                with open(output_file, 'w') as file:
                    file.write(response_text)
                print(f"Saved response for video ID {video_id} to {output_file}")

            except Exception as e:
                with open(issue_file, 'a') as file:
                    file.write(f"{url}\n")
                    print(f"Failed to process video ID {video_id}: {e}")
        else:
            # Log the unavailable video URL
            with open(unavailable_file, 'a') as file:
                file.write(f"{url}\n")
            print(f"Video not available in storage: {url}")

    else:
        print(f"Invalid YouTube URL: {url}")
