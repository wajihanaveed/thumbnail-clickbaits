import base64
import os
import pandas as pd
from anthropic import AnthropicVertex
import time

# Set your location and project ID
LOCATION = "europe-west1"
PROJECT_ID = "your-project-id"

# Initialize the Anthropic client
client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)


csv_file_path = "/path/to/your/mtv-csv/"
thumbnails_folder = "MTV_Thumbnails"
output_folder = "MTV_Thumbnail_Description"

# Repeat for NMTV

# csv_file_path = "/path/to/your/nmtv-csv/"
# thumbnails_folder = "NMTV_Thumbnails"
# output_folder = "NMTV_Thumbnail_Description"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Helper function to extract YouTube video ID from URL
def extract_video_id(url):
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[-1]
    elif "youtu.be/" in url:
        return url.split("/")[-1]
    return None

# Iterate through the DataFrame and process each URL
for index, row in df.iterrows():
    time.sleep(10)
    url = row['url']
    video_id = extract_video_id(url)

    if video_id:
        thumbnail_path = os.path.join(thumbnails_folder, f"{video_id}.jpg")
        
        # Check if the thumbnail exists
        if os.path.exists(thumbnail_path):
            # Read and encode the thumbnail image
            with open(thumbnail_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            image_media_type = "image/jpeg"
            
            # Send the image to Anthropic to get the description
            message = client.messages.create(
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Describe this thumbnail in one sentence"
                            }
                        ],
                    }
                ],
                model="claude-3-5-sonnet@20240620",
            )
            
            # Extract the description from the 'content' list
            if message.content and isinstance(message.content, list):
                description = ""
                for block in message.content:
                    if block.type == "text":
                        description = block.text
                        break  
            else:
                description = "No description available."
            
            # Save the description to a text file
            output_file_path = os.path.join(output_folder, f"{video_id}.txt")
            with open(output_file_path, "w") as text_file:
                text_file.write(description)
            
            print(f"Processed {video_id}")
        else:
            print(f"Thumbnail for video ID {video_id} not found.")
    else:
        print(f"Invalid YouTube URL: {url}")
