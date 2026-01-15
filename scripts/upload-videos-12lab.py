import os
import csv
import subprocess
from pytubefix import YouTube
from pytubefix.exceptions import RegexMatchError, VideoUnavailable
from twelvelabs import TwelveLabs
from shlex import quote  

API_KEY = "<api-key>"
INDEX_ID = "<index-id>"

client = TwelveLabs(api_key=API_KEY)
def download_video(url):
    try:
        yt = YouTube(url)
        video_id = yt.video_id
        
        # Filter video streams with a resolution of 360p or greater
        video_streams = yt.streams.filter(progressive=False, file_extension='mp4', res='360p').order_by('resolution')
        
        if not video_streams:
            print("No video streams available with 360p or greater resolution.")
            return None
        
        print(f"\nAvailable video resolutions for {yt.title}:")
        for stream in video_streams:
            print(f"Resolution: {stream.resolution}")
        
        best_video_stream = video_streams.desc().first()
        

        audio_streams = yt.streams.filter(only_audio=True).order_by('abr')
        if not audio_streams:
            print("No audio streams available.")
            return None
        
        print(f"\nAvailable audio bitrates for {yt.title}:")
        for stream in audio_streams:
            print(f"Audio Bitrate: {stream.abr}")
        
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



def upload_12labs(file_name):
    try:
        task = client.task.create(
            index_id=INDEX_ID,
            file=file_name,  
        )
        
        print(f"Task id={task.id}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def trim_video_if_necessary(file_path, max_duration="29:55"):
    try:
        # Quote the file path to handle special characters
        quoted_file_path = quote(file_path)

        # Get the duration of the video
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", quoted_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout)

        # 30 minutes in seconds
        max_duration_seconds = 30 * 60
        
        if duration > max_duration_seconds:
            temp_file_path = file_path.replace(".mp4", "_temp.mp4")
            # Trim the video to 29 minutes and 55 seconds and save to a temporary file
            subprocess.run(
                ["ffmpeg", "-y", "-i", quote(file_path), "-ss", "00:00:00", "-t", max_duration, "-c", "copy", quote(temp_file_path)]
            )
            print(f"Video trimmed to {max_duration} minutes. Replacing original file.")
            os.replace(temp_file_path, file_path)
        else:
            print("Video duration is within the limit, no trimming necessary.")
        
        return file_path
    except Exception as e:
        print(f"An error occurred while trimming the video: {e}")
        return None

def delete_local_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")
    except OSError as e:
        print(f"Error deleting file: {e}")


# # Path to your CSV file
csv_file_path = "/path/to/your/csv/"

# # Process URLs from CSV
save_path = "/path/to/your/cwd/"

with open(csv_file_path, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        url = row.get('url')
        if url:
            video_filename = download_video(url)
            if video_filename:
                local_file_path = os.path.join(save_path, video_filename)
                
                trimmed_file_path = trim_video_if_necessary(local_file_path)
                if trimmed_file_path and upload_12labs(trimmed_file_path):
                    delete_local_file(trimmed_file_path)
