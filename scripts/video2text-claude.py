import csv
import os
import cv2
import json
import base64
import subprocess
from pytubefix import YouTube
from pytubefix.exceptions import RegexMatchError, VideoUnavailable
from anthropic import AnthropicVertex
from shlex import quote  # For safely quoting file names

# Set your location and project ID
LOCATION = "europe-west1"
PROJECT_ID = "your-project-id"
client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

# Constants
# RESULTS_DIR = "Claude_NMTV_Video_To_Text"
RESULTS_DIR = "Claude_MTV_Video_To_Text"
UNAVAILABLE_FILE = "claude_video_to_text_unavailable.txt"
FAILED_FILE = "claude_video_to_text_failed.txt"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to download video 
def download_video(url):
    try:
        yt = YouTube(url)
        video_id = yt.video_id
        
        # Filter video streams with a resolution of 360p or greater
        video_streams = yt.streams.filter(progressive=False, file_extension='mp4', res='360p').order_by('resolution')
        
        if not video_streams:
            print("No video streams available with 360p or greater resolution.")
            return None
        
        # print(f"\nAvailable video resolutions for {yt.title}:")
        # for stream in video_streams:
        #     print(f"Resolution: {stream.resolution}")
        
        best_video_stream = video_streams.desc().first()
        
        # Filter audio streams
        audio_streams = yt.streams.filter(only_audio=True).order_by('abr')
        if not audio_streams:
            print("No audio streams available.")
            return None
        
        # print(f"\nAvailable audio bitrates for {yt.title}:")
        # for stream in audio_streams:
        #     print(f"Audio Bitrate: {stream.abr}")
        
        best_audio_stream = audio_streams.desc().first()
        
        # Download video and audio
        print(f"\nDownloading video: {yt.title}")
        print(f"Video Resolution: {best_video_stream.resolution}")
        video_file = best_video_stream.download(filename=f'{quote(video_id)}_video.mp4')
        
        print(f"\nDownloading audio: {yt.title}")
        print(f"Audio Bitrate: {best_audio_stream.abr}")
        audio_file = best_audio_stream.download(filename=f'{quote(video_id)}_audio.mp3')
        
        print("Download completed! Now merging video and audio...")
        
        # Merge video and audio using ffmpeg
        video_file_path = os.path.abspath(video_file)
        audio_file_path = os.path.abspath(audio_file)
        output_file = os.path.abspath(f'{video_id}.mp4')
        
        merge_command = f'ffmpeg -i {quote(video_file_path)} -i {quote(audio_file_path)} -c:v copy -c:a aac {quote(output_file)}'
        subprocess.run(merge_command, shell=True)
        
        print(f"Merging completed! The final video is saved as {output_file}")
        
        # Cleanup
        delete_local_file(video_file_path)
        delete_local_file(audio_file_path)
        
        return output_file
        
    except RegexMatchError as e:
        print(f"Error: {e}. The URL provided is not valid.")
        return None 
    except VideoUnavailable as e:
        print(f"Error: {e}. The video is unavailable.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Function to extract frames from video
def extract_frames(video_path, frame_count=20):
    video = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // frame_count

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if len(frames) < frame_count:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        else:
            break
        video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_POS_FRAMES) + interval)

    video.release()
    return frames

# Function to generate description using Claude
def generate_description(model_name, frames):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame,
                    },
                } for frame in frames
            ] + [
                {
                    "type": "text",
                    "text": "Consider these frames as continuous scenes from a video. Provide a detailed description of the video content, breaking it down scene by scene. Focus on key actions, visuals, emotions, and any notable details. Describe it as if you are watching the full video, ensuring that the narrative is cohesive and captures the flow of the scenes.",
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

# Main function to process videos
def process_videos(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)  # Use DictReader to access specific columns
        for row in reader:
            video_url = row['url']  # The column name in the CSV is 'url'
            video_id = video_url.split('v=')[-1]
            destination_file_name = f"{video_id}.mp4"
            
            if not download_video(video_url):
                with open(UNAVAILABLE_FILE, 'a') as uf:
                    uf.write(video_url + '\n')
                continue

            try:
                frames = extract_frames(destination_file_name)
                if not frames:
                    raise ValueError("No frames extracted")
                
                for model in ["claude-3-5-sonnet@20240620"]:
                    description = generate_description(model, frames)

                    text_content = ""
                    for block in description.content:
                        if block.type == 'text':
                            text_content += block.text + "\n"

                    output_path = os.path.join(RESULTS_DIR, f"{video_id}.txt")

                    # Write the extracted text to the output file
                    with open(output_path, 'w') as output_file:
                        output_file.write(text_content)

                    print(f"Processed video: {video_id}")

            except Exception as e:
                with open(FAILED_FILE, 'a') as ff:
                    ff.write(video_url + '\n')
                print(f"Failed to process video {video_id}: {e}")

            finally:
                if os.path.exists(destination_file_name):
                    os.remove(destination_file_name)
                print(f"Deleted local file: {destination_file_name}")


def delete_local_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")
    except OSError as e:
        print(f"Error deleting file: {e}")

# Load the CSV file
csv_file = "/path/to/your/csv/"
process_videos(csv_file)
