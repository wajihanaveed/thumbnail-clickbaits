import os
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import time
# Initialize Vertex AI and Google Cloud Storage
vertexai.init(project="your-project-id", location="us-east1")
storage_client = storage.Client()

# Load the CSV file with URLs
csv_file = "/path/to/your/csv/"
df = pd.read_csv(csv_file)

# Initialize the Gemini model
vision_model = GenerativeModel("gemini-1.5-flash-001")

'''For ease we will be evaluating NMTVs and NMTVs separately, uncomment the next two lines and comment the next two to evaluate NMTVs'''
# description_dir = 'Gemini_NMTV_Video_To_Text'
# subtitles_dir = 'NMTV_Subtitles'
# output_dir = 'Gemini_NMTV_Noexample_Results'
description_dir = 'Gemini_MTV_Video_To_Text'
subtitles_dir = 'MTV_Subtitles'
output_dir = 'Gemini_MTV_Noexample_Results'
os.makedirs(output_dir, exist_ok=True)

# Define unavailable and issue files to be stored in the current working directory (CWD)
unavailable_file = os.path.join(os.getcwd(), 'unavailable.txt')
issue_file = os.path.join(os.getcwd(), 'issue.txt')

# Function to extract the video ID from a YouTube URL
def extract_video_id(url):
    if 'youtube.com/watch?v=' in url:
        return url.split('youtube.com/watch?v=')[-1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    else:
        return None
def check_thumbnail_exists(bucket_name, thumbnail_blob_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(thumbnail_blob_name)
        return blob.exists()  # Check if the file exists in the bucket
    except Exception as e:
        print(f"Error checking thumbnail existence: {e}")
        return False
    
# Function to check the availability of necessary files for a given video ID
def check_required_files(video_id):
    # Thumbnail is stored in Google Cloud Storage, no need to check locally
    description_path = os.path.join(description_dir, f"{video_id}.txt")
    subtitles_path = os.path.join(subtitles_dir, f"{video_id}.txt")

    return (
        os.path.exists(description_path),
        os.path.exists(subtitles_path),
        description_path,
        subtitles_path
    )

# Iterate over the URLs in the CSV
for url in df['url']:
    time.sleep(22)
    video_id = extract_video_id(url)
    if video_id:
        description_exists, subtitles_exists, description_path, subtitles_path = check_required_files(video_id)

        # Define the Cloud Storage thumbnail path
        thumbnail_path = f"gs://<bucket-name>/{video_id}.jpg"
        # Define the Cloud Storage thumbnail path and bucket details
        bucket_name = "<bucket-name>"
        thumbnail_blob_name = f"{video_id}.jpg"
        
        # Check if the thumbnail exists in the Google Cloud Storage bucket
        thumbnail_exists = check_thumbnail_exists(bucket_name, thumbnail_blob_name)
        
        if description_exists and thumbnail_exists:
            
            try:
                # Read the description
                with open(description_path, 'r') as desc_file:
                    video_description = desc_file.read()
                
                # Handle subtitles availability
                if subtitles_exists:
                    with open(subtitles_path, 'r') as subtitles_file:
                        video_subtitles = subtitles_file.read()
                else:
                    video_subtitles = "Subtitles not available."
                
                # Prepare the prompt for the Gemini model
                prompt = f"""
                Task: Analyze the provided information about a YouTube video and determine whether its thumbnail is misleading or not misleading. You will be given the following information:
                        1. The video's thumbnail
                        2. The video's subtitles
                        3. A text description of the video content
                        Steps to follow:
                        1. Carefully examine the thumbnail.
                        2. Read through the video subtitles and content description.
                        3. Compare the thumbnail to the actual video content.
                        4. Determine if the thumbnail accurately represents the video's main topic or content.
                        5. Assess whether the thumbnail uses exaggeration, false promises, or clickbait tactics.
                        6. Categorize the thumbnail as either "Misleading" or "Not Misleading".
                        7. Provide a brief explanation for your decision.
                        
                        Analyze the provided information and categorize the thumbnail as "Misleading" or "Not Misleading", followed by a brief explanation for your decision.
                Video Description:
                {video_description}

                Video Subtitles:
                {video_subtitles}
                """

                # Send the prompt to the Gemini model along with the video, thumbnail, and prompt
                response = vision_model.generate_content(
                    [
                        Part.from_uri(thumbnail_path, mime_type="image/jpeg"),  # Thumbnail
                        prompt  # Custom prompt with video description and subtitles
                    ]
                )

                # Save the result
                output_file = os.path.join(output_dir, f"{video_id}.txt")
                with open(output_file, 'w') as file:
                    file.write(response.text)
                print(f"Saved response for video ID {video_id} to {output_file}")

            except Exception as e:
                # Log issues in processing
                with open(issue_file, 'a') as file:
                    file.write(f"Error processing video ID {video_id}: {e}\n")
                print(f"Failed to process video ID {video_id}: {e}")
        else:
            # Log unavailable resources (description or subtitles)
            with open(unavailable_file, 'a') as file:
                file.write(f"Missing resources for video ID {video_id}: Description: {description_exists}, Subtitles: {subtitles_exists}\n")
            print(f"Missing resources for video ID {video_id}: Description: {description_exists}, Subtitles: {subtitles_exists}")
    else:
        print(f"Invalid YouTube URL: {url}")
