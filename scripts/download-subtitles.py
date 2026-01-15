import os
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from urllib.parse import urlparse, parse_qs

def get_video_id(youtube_url):
    """
    Extract the video ID from a YouTube URL.
    """
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]  
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]  
    return None

def get_any_transcript(video_id, limit_duration=1795):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None

        for transcript_obj in transcript_list:
            if not transcript_obj.is_generated:
                transcript = transcript_obj.fetch()
                break
        if transcript is None:
            for transcript_obj in transcript_list:
                if transcript_obj.is_generated:
                    transcript = transcript_obj.fetch()
                    print(f"Using auto-generated transcript in {transcript_obj.language}")
                    break
        
        if transcript is None:
            print(f"No transcript available for video ID {video_id}")
            return None
        

        limited_transcript = [entry for entry in transcript if entry['start'] <= limit_duration]
        return limited_transcript
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def translate_transcript(transcript, dest_lang='en'):
    """
    Translate the transcript into a specified language.
    """
    translator = GoogleTranslator(target=dest_lang)
    translated_transcript = []
    for entry in transcript:
        try:
            if entry['text'] is not None:
                translated_text = translator.translate(entry['text'])
                translated_transcript.append({'text': translated_text, 'start': entry['start'], 'duration': entry['duration']})
        except Exception as e:
            print(f"An error occurred during translation: {e}")
    return translated_transcript

def save_transcript_to_file(video_id, transcript, output_folder):
    """
    Save the transcript to a text file.
    """
    file_path = os.path.join(output_folder, f"{video_id}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in transcript:
            if entry['text'] is not None:
                f.write(entry['text'] + "\n")
    print(f"Saved transcript for video ID {video_id} to {file_path}")

def main(csv_file, output_folder, translate=True, no_subtitles_csv='no-subtitles.csv', limit_duration=1795):
    """
    Main function to process each YouTube URL, fetch its transcript (limited to 29mins 55 sec), optionally translate, and save to a file.
    If no transcript is available, save the URL to no-subtitles.csv.
    """

    df = pd.read_csv(csv_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    no_subtitle_urls = []

    for youtube_url in df['url']:
        video_id = get_video_id(youtube_url)
        if video_id:
            print(f"Processing video ID: {video_id}")
            transcript = get_any_transcript(video_id, limit_duration=limit_duration)
            if transcript:
                if translate:
                    transcript = translate_transcript(transcript, dest_lang='en')
                save_transcript_to_file(video_id, transcript, output_folder)
            else:
                print(f"No transcript available for video ID: {video_id}")
                no_subtitle_urls.append(youtube_url)  
        else:
            print(f"Invalid YouTube URL: {youtube_url}")
            no_subtitle_urls.append(youtube_url)

    if no_subtitle_urls:
        no_subtitle_df = pd.DataFrame(no_subtitle_urls, columns=['url'])
        no_subtitle_df.to_csv(no_subtitles_csv, index=False)
        print(f"Saved URLs of videos with no subtitles to {no_subtitles_csv}")

if __name__ == "__main__":

    """For ease in evaluation/testing store the MTV and NMTV urls in separate csvs and name the output folder as MTV_Subtitles or NMTV_Subtitles"""
    csv_file = "/path/to/your/csv/"
    # output_folder = "NMTV_Subtitles"
    output_folder = "MTV_Subtitles"

    main(csv_file, output_folder, limit_duration=1795)
