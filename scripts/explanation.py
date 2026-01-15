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

# Explanations only used for dynamic few-shot prompting 



csv_file = "/path/to/your/csv/"
df = pd.read_csv(csv_file)

"""For NMTV explnations"""
# subtitles_dir = 'NMTV_Subtitles'
# thumbnails_dir = 'NMTV_Thumbnail_Description'
# description_dir = 'Claude_NMTV_Video_To_Text'
# results_dir = 'NMTV_Explanations'

# Generating MTV Explanations

df = pd.read_csv(csv_file)
subtitles_dir = 'MTV_Subtitles'
thumbnails_dir = 'MTV_Thumbnail_Description'
description_dir = 'Claude_MTV_Video_To_Text'
results_dir = 'MTV_Explanations'


os.makedirs(results_dir, exist_ok=True)

# Helper function to truncate text to 200 words
def truncate_text(text, max_words=200):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text

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
    thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}.txt")
    
    if os.path.exists(description_path) and os.path.exists(thumbnail_path):
        # Read subtitle, use default message if not available
        if os.path.exists(subtitle_path):
            with open(subtitle_path, 'r') as s_file:
                subtitles = s_file.read()

        else:
            subtitles = "Subtitles not available."
        
        with open(description_path, 'r') as d_file:
            description = d_file.read()

        
        with open(thumbnail_path, 'r') as d_file:
            thumbnail = d_file.read()

        
        return subtitles, description, thumbnail
    else:
        return None, None, None

# Function to prepare the prompt based on video description, subtitles, and thumbnail
def prepare_prompt(video_description, video_subtitles, thumbnail):
    # Truncate the description and subtitles to 200 words
    truncated_description = truncate_text(video_description, max_words=200)
    truncated_subtitles = truncate_text(video_subtitles, max_words=200)
    
    prompt = f"""
        Task: A thumbnail is considered misleading if its content is inaccurate or shows exaggeration, false promises, or clickbait tactics else it is considered Not Misleading. You are given a YouTube video's textual thumbnail description, its categorization as Misleading or Not Misleading, the video's content as video to text description and subtitles, write a one sentence explanation justifying the categorization of that thumbnail. 
    Thumbnail Description:
    {thumbnail}
    Categorization: Not Misleading
    Video Description:
    {truncated_description}
    Video Subtitles:
    {truncated_subtitles}
    """
    return prompt

# Function to send the prompt and data to Claude
def send_to_claude(model_name, prompt, thumbnail):
    prompt_messages = [
        {
            "role": "user",
            "content": [

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

        subtitles, description, thumbnail = retrieve_files(video_id)

        
        if subtitles and description and thumbnail:
            prompt_text = prepare_prompt(description, subtitles, thumbnail)
            
            # Send prompt and thumbnail to Claude for each model
            for model in models:
                try:
                    print(f"Model: {model}")
                    description_response = send_to_claude(model, prompt_text, thumbnail)
                    
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
                    print(f"Error with model {model}, couldn't process{video_id}: {e}")
        else:
            print(f"Missing files for video ID {video_id}")
    else:
        print(f"Invalid YouTube URL: {url}")
