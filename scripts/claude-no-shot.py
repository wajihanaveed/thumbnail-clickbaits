import os
import pandas as pd
from anthropic import AnthropicVertex
import base64
import json
import time

# Set your location and project ID
LOCATION = "europe-west1"
PROJECT_ID = "your-project-id"
client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

# Load the CSV file containing YouTube URLs
csv_file = "/path/to/your/csv/"
df = pd.read_csv(csv_file)

# Directories for subtitles, thumbnails, descriptions, and results
'''For ease we will be evaluating NMTVs and NMTVs separately, uncomment the next four lines and comment the next 4 to evaluate NMTVs'''
# subtitles_dir = 'NMTV_Subtitles'
# thumbnails_dir = 'NMTV_Thumbnails'
# description_dir = 'Claude_NMTV_Video_To_Text'
# results_dir = 'Claude_NMTV_Noexample_Results'

subtitles_dir = 'MTV_Subtitles'
thumbnails_dir = 'MTV_Thumbnails'
description_dir = 'Claude_MTV_Video_To_Text'
results_dir = 'Claude_MTV_Noexample_Results'
os.makedirs(results_dir, exist_ok=True)

# Function to extract the video ID from a YouTube URL
def extract_video_id(url):
    if 'youtube.com/watch?v=' in url:
        return url.split('youtube.com/watch?v=')[-1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    else:
        return None

# Function to retrieve necessary files for a given video ID
def retrieve_files(video_id):
    subtitle_path = os.path.join(subtitles_dir, f"{video_id}.txt")
    description_path = os.path.join(description_dir, f"{video_id}.txt")
    thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}.jpg")
    
    if os.path.exists(description_path) and os.path.exists(thumbnail_path):
        # Read subtitle, use default message if not available
        if os.path.exists(subtitle_path):
            with open(subtitle_path, 'r') as s_file:
                subtitles = s_file.read()
        else:
            subtitles = "Subtitles not available."
        
        with open(description_path, 'r') as d_file:
            description = d_file.read()
        
        with open(thumbnail_path, 'rb') as thumb_file:
            thumbnail_base64 = base64.b64encode(thumb_file.read()).decode("utf-8")
        
        return subtitles, description, thumbnail_base64
    else:
        return None, None, None

# Function to prepare the prompt based on video description, subtitles, and thumbnail
def prepare_prompt(video_description, video_subtitles):
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
    return prompt

# Function to send the prompt and data to Claude
def send_to_claude(model_name, prompt, thumbnail_base64):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": thumbnail_base64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    response = client.messages.create(
        max_tokens=4096,
        messages=prompt_messages,
        model=model_name,
    )
    
    return response

# Model list to test
models = ["claude-3-5-sonnet@20240620"]
# Iterate over the URLs in the CSV and generate the prompt
for url in df['url']:
    time.sleep(10)
    video_id = extract_video_id(url)
    
    if video_id:
        subtitles, description, thumbnail_base64 = retrieve_files(video_id)
        
        if subtitles and description and thumbnail_base64:
            prompt_text = prepare_prompt(description, subtitles)
            
            # Send prompt and thumbnail to Claude for each model
            for model in models:
                try:
                    print(f"Model: {model}")
                    description_response = send_to_claude(model, prompt_text, thumbnail_base64)
                    
                    # Accumulate text content from the response
                    text_content = ""
                    for block in description_response.content:  # Use dot notation instead of brackets
                        if block.type == 'text':  # Use dot notation for 'type' and 'text'
                            text_content += block.text + "\n"  # Accumulate all text blocks

                    # Save the text content to a file
                    output_file = os.path.join(results_dir, f"{video_id}.txt")
                    with open(output_file, 'w') as f:
                        f.write(text_content)  # Save only text content
                    print(f"Saved response for video ID {video_id} to {output_file}")
                    
                except Exception as e:
                    print(f"Error with model {model}: {e}")
        else:
            print(f"Missing files for video ID {video_id}")
    else:
        print(f"Invalid YouTube URL: {url}")
