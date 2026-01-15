import base64
import requests
import pandas as pd
import os

# OpenAI API Key
api_key = "<api-key>"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to extract video ID from a YouTube URL
def extract_video_id(url):
    # This will handle standard YouTube links
    if 'youtube.com/watch?v=' in url:
        return url.split('watch?v=')[1][:11]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1][:11]
    return None

# Function to classify if a thumbnail is misleading or not
def classify_thumbnail(image_path, videoid_text, subtitle_text):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # For GPT 4o replace model with "model": "gpt-4o-2024-05-13"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""

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
                            Examples:
                            Example 1:
                            Thumbnail: A person holding a stack of $100 bills with the text "I made $10,000 in one day!"
                            Subtitles: "In this video, I'll share my experience of how I earned $500 in a week through freelancing."
                            Video description: The creator discusses various freelancing opportunities and how to get started, sharing their personal experience of earning $500 in their first week.
                            Categorization: Misleading
                            Explanation: The thumbnail exaggerates the earnings ($10,000 in one day) compared to the actual content ($500 in a week), using clickbait tactics to attract viewers.

                            Example 2:
                            Thumbnail: A smiling chef holding a plate of pasta with the text "Easy 15-minute pasta recipe"
                            Subtitles: "Today, we're making a quick and delicious pasta dish that takes only 15 minutes to prepare."
                            Video description: The video demonstrates a step-by-step process of cooking a simple pasta dish, timing the preparation to show it takes approximately 15 minutes.
                            Categorization: Not Misleading
                            Explanation: The thumbnail accurately represents the video content, showing the final dish and correctly stating the preparation time.
                            Now, please analyze the provided information and categorize the thumbnail as "Misleading" or "Not Misleading", followed by a brief explanation for your decision.

                                Video Description:
                                {videoid_text}

                                Video Subtitles:
                                {subtitle_text}
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Extract the classification result (just the content of the first choice)
    classification_result = response.json()["choices"][0]["message"]["content"]

    return classification_result

# Path to CSV file containing YouTube URLs
csv_file_path = "/path/to/your/csv/"

# Path to folders
# thumbnail_folder = "NMTV_Thumbnails"
# subtitles_folder = "NMTV_Subtitles"
# desc_folder = "12Lab_NMTV_Video_To_Text"
# results_folder = "GPT4_NMTV_FixedShot_Results"
thumbnail_folder = "MTV_Thumbnails"
subtitles_folder = "MTV_Subtitles"
desc_folder = "12Lab_MTV_Video_To_Text"
results_folder = "GPT4_MTV_FixedShot_Results"

# Ensure the results folder exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Iterate over each URL in the CSV file
for url in df['url']:
    video_id = extract_video_id(url)

    if video_id:
        # Paths for the current video's resources
        thumbnail_path = os.path.join(thumbnail_folder, f"{video_id}.jpg")
        subtitles_path = os.path.join(subtitles_folder, f"{video_id}.txt")
        desc_path = os.path.join(desc_folder, f"{video_id}.txt")
        result_file_path = os.path.join(results_folder, f"{video_id}.txt")

        # Check if thumbnail and description exist, skip if not
        if not os.path.exists(thumbnail_path) or not os.path.exists(desc_path):
            print(f"Skipping {video_id}: Thumbnail or description not available.")
            continue
        # Check if subtitles exist
        if os.path.exists(subtitles_path):
            # Read videoid.txt and subtitles.txt content
            with open(subtitles_path, 'r') as subtitle_file:
                subtitle_text = subtitle_file.read()
        else:
            subtitle_text = "Subtitles not available."
            print(f"Subtitles not availble for {video_id}.")

        # Assuming videoid.txt is stored in the description folder
        with open(desc_path, 'r') as videoid_file:
            videoid_text = videoid_file.read()

        # Classify thumbnail with additional context
        classification_result = classify_thumbnail(thumbnail_path, videoid_text, subtitle_text)
        
        # Store the result in a text file inside the results folder
        with open(result_file_path, 'w') as result_file:
            result_file.write(f"Classification result for {video_id}:\n")
            result_file.write(f"{classification_result}\n")
            result_file.write("\n----------------------------------------\n")
        print(f"Classification result stored for {video_id} in {result_file_path}")
