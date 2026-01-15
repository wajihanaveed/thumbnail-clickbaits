import os
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import warnings

# **Configuration and Initialization**

# Initialize Vertex AI and Google Cloud Storage
vertexai.init(project="your-project-id", location="us-east1")
storage_client = storage.Client()

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.",
    category=FutureWarning,
    module='transformers.tokenization_utils_base'
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the CSV file containing YouTube URLs
csv_file = 'path/to/your/csv'
try:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV file '{csv_file}' successfully.")

except FileNotFoundError:
    print(f"Error: CSV file '{csv_file}' not found.")
    raise
except pd.errors.EmptyDataError:
    print(f"Error: CSV file '{csv_file}' is empty.")
    raise
except pd.errors.ParserError:
    print(f"Error: CSV file '{csv_file}' is malformed.")
    raise

# Directories for subtitles, descriptions, and results
'''For ease we will be evaluating NMTVs and NMTVs separately, uncomment the next three lines and comment the next 3 to evaluate NMTVs'''
# subtitles_dir = 'NMTV_Subtitles'
# description_dir = 'Gemini_NMTV_Video_To_Text'
# results_dir = 'Dynamic_NMTV_Gemini'
subtitles_dir = 'MTV_Subtitles'
description_dir = 'Gemini_MTV_Video_To_Text'
results_dir = 'Dynamic_MTV_Gemini'
os.makedirs(results_dir, exist_ok=True)


similar_thumbnails_thumb_desc_dir_mtv = 'MTV_Thumbnail_Description'
similar_thumbnails_thumb_desc_dir_non_mtv = 'NMTV_Thumbnail_Description'
similar_video_explanations_dir_mtv = 'MTV_Explanations'
similar_video_explanations_dir_non_mtv = 'NMTV_Explanations'
similar_subtitles_dir_mtv = 'MTV_Subtitles'
similar_subtitles_dir_non_mtv = 'NMTV_Subtitles'
similar_description_dir_mtv = 'Gemini_MTV_Video_To_Text'
similar_description_dir_non_mtv = 'Gemini_NMTV_Video_To_Text'

# Initialize the Sentence-BERT model once
try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("Sentence-BERT model loaded successfully.")
except Exception as e:
    print(f"Failed to load Sentence-BERT model: {e}")
    raise

# Load and preprocess CSV once
data_csv = 'dynamic.csv'
try:
    data = pd.read_csv(data_csv)
    required_columns = {'Video ID', 'Video to Text Description', 'Label'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"The CSV file '{data_csv}' is missing required columns: {missing}")
    data['Video to Text Description'] = data['Video to Text Description'].fillna('')
    label_mapping = data.set_index('Video ID')['Label'].to_dict()
    print(f"Loaded and preprocessed CSV file '{data_csv}' successfully.")
except FileNotFoundError:
    print(f"Error: CSV file '{data_csv}' not found.")
    raise
except pd.errors.EmptyDataError:
    print(f"Error: CSV file '{data_csv}' is empty.")
    raise
except pd.errors.ParserError:
    print(f"Error: CSV file '{data_csv}' is malformed.")
    raise
except Exception as e:
    print(f"An error occurred while loading '{data_csv}': {e}")
    raise

# Optional: Precompute and cache video embeddings to improve performance
# Uncomment the following lines if you wish to implement caching
import pickle
embeddings_file = 'video_embeddings.pkl'
if os.path.exists(embeddings_file):
    try:
        with open(embeddings_file, 'rb') as f:
            video_embeddings = pickle.load(f)
            print(f"Loaded precomputed video embeddings from '{embeddings_file}'.")
    except Exception as e:
        print(f"Failed to load embeddings from '{embeddings_file}': {e}")
        raise
else:
    try:
        video_embeddings = model.encode(data['Video to Text Description'].tolist(), batch_size=64, show_progress_bar=True)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(video_embeddings, f)
            print(f"Computed and saved video embeddings to '{embeddings_file}'.")
    except Exception as e:
        print(f"Failed to compute video embeddings: {e}")
        raise

# Initialize the Gemini model
vision_model = GenerativeModel("gemini-1.5-flash-001")



def extract_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    
    Parameters:
    - url (str): The YouTube URL.
    
    Returns:
    - str or None: The extracted video ID or None if extraction fails.
    """
    if 'youtube.com/watch?v=' in url:
        return url.split('youtube.com/watch?v=')[-1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    else:
        return None

def check_thumbnail_exists(bucket_name, thumbnail_blob_name):
    """
    Checks if a thumbnail exists in the specified Google Cloud Storage bucket.
    
    Parameters:
    - bucket_name (str): The name of the GCS bucket.
    - thumbnail_blob_name (str): The name of the thumbnail blob.
    
    Returns:
    - bool: True if exists, False otherwise.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(thumbnail_blob_name)
        return blob.exists()
    except Exception:
        return False

def check_required_files(video_id):
    """
    Checks the availability of required files (description and subtitles) for a video.
    
    Parameters:
    - video_id (str): The YouTube video ID.
    
    Returns:
    - tuple: (description_exists, subtitles_exists, description_path, subtitles_path)
    """
    description_path = os.path.join(description_dir, f"{video_id}.txt")
    subtitles_path = os.path.join(subtitles_dir, f"{video_id}.txt")

    
    description_exists = os.path.exists(description_path)
    subtitles_exists = os.path.exists(subtitles_path)
    
    return (description_exists, subtitles_exists, description_path, subtitles_path)

def retrieve_files(video_id):
    """
    Retrieves the description and subtitles for a given video ID.
    
    Parameters:
    - video_id (str): The YouTube video ID.
    
    Returns:
    - tuple: (description, subtitles)
    """
    description_path = os.path.join(description_dir, f"{video_id}.txt")
    subtitles_path = os.path.join(subtitles_dir, f"{video_id}.txt")
    
    try:
        with open(description_path, 'r', encoding='utf-8') as desc_file:
            description = desc_file.read()
    except Exception:
        description = ""
    
    if os.path.exists(subtitles_path):
        try:
            with open(subtitles_path, 'r', encoding='utf-8') as subtitles_file:
                subtitles = subtitles_file.read()
        except Exception:
            subtitles = "Subtitles not available."
    else:
        subtitles = "Subtitles not available."
    
    return description, subtitles

def prepare_prompt(video_description, video_subtitles, similar_examples=None):
    """
    Prepares the prompt for the AI model, including any similar video examples.
    
    Parameters:
    - video_description (str): Description of the input video.
    - video_subtitles (str): Subtitles of the input video.
    - similar_examples (list of dict): List containing similar video examples.
    
    Returns:
    - str: The formatted prompt.
    """
    base_prompt = f"""
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
"""
    
    # Add similar examples if provided
    if similar_examples:
        for idx, example in enumerate(similar_examples, start=2):  # Start from 2 since Image 1 is main thumbnail
            truncated_subtitles = truncate_text(example['subtitles'], max_words=200)
            truncated_description = truncate_text(example['description'], max_words=200)
            base_prompt += f"""

Example {idx - 1}:
Thumbnail: {example['thumbnail_description']}
Subtitles: {truncated_subtitles}
Video Description: {truncated_description}
Categorization: {example['categorization']}
Explanation: {example['explanation']}
"""
    
    # Add the current video information
    base_prompt += f"""

Now, please analyze the provided information and categorize the main video's thumbnail as "Misleading" or "Not Misleading", followed by a brief explanation for your decision.

Video Description:
{video_description}

Video Subtitles:
{video_subtitles}
"""
    return base_prompt


def get_top_similar_videos(
    video_description: str,
    video_id: str,
    data: pd.DataFrame,
    model: SentenceTransformer,
    top_n: int = 100
) -> List[Dict[str, str]]:
    """
    Finds the top N most similar videos to the given video description and ID using Sentence-BERT.

    Parameters:
    - video_description (str): Description of the video to find similarities for.
    - video_id (str): The ID of the video.
    - data (pd.DataFrame): DataFrame containing video data with at least 'Video ID', 'Video to Text Description', and 'Label' columns.
    - model (SentenceTransformer): Pre-initialized Sentence-BERT model for encoding descriptions.
    - top_n (int): Number of top similar videos to retrieve.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries containing similar video IDs, their labels, and similarity scores.
    """
    try:
        # Validate DataFrame columns
        required_columns = {'Video ID', 'Video to Text Description', 'Label'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            print(f"The DataFrame is missing required columns: {missing}")
            return []
        
        # Convert video descriptions to sentence embeddings
        video_embeddings = model.encode(data['Video to Text Description'].tolist())
        input_embedding = model.encode([video_description])
        
        # Compute cosine similarity between input video description and other videos
        cosine_sim = cosine_similarity(input_embedding, video_embeddings).flatten()
        
        # Create a DataFrame for similarity scores
        similarity_df = pd.DataFrame({
            'Video ID': data['Video ID'],
            'Label': data['Label'],
            'Similarity Score': cosine_sim
        })
        
        # Remove the input video from the similarity results
        similarity_df = similarity_df[similarity_df['Video ID'] != video_id]
        
        # Sort the DataFrame by similarity score in descending order
        similarity_df = similarity_df.sort_values(by='Similarity Score', ascending=False)
        
        # Select the top N similar videos
        top_similar = similarity_df.head(top_n)
        
        # Convert the results to a list of dictionaries
        top_similar_list = top_similar.to_dict('records')
        return top_similar_list
    
    except Exception as e:
        print(f"An error occurred in get_top_similar_videos: {e}")
        return []

def retrieve_similar_video_data(similar_video_id, label):
    """
    Retrieves the thumbnail description, subtitles, and description for a similar video based on its label.

    Parameters:
    - similar_video_id (str): The ID of the similar video.
    - label (str): The label of the similar video ('MTV' or 'Non MTV').

    Returns:
    - Dict: A dictionary containing 'thumbnail_description', 'subtitles', 'description', and 'explanation'.
    """
    try:
        if label == 'MTV':
            thumbnail_desc_dir = similar_thumbnails_thumb_desc_dir_mtv
            explanation_dir = similar_video_explanations_dir_mtv
            subtitles_dir_sim = similar_subtitles_dir_mtv
            description_dir_sim = similar_description_dir_mtv
        elif label == 'Non MTV':
            thumbnail_desc_dir = similar_thumbnails_thumb_desc_dir_non_mtv
            explanation_dir = similar_video_explanations_dir_non_mtv
            subtitles_dir_sim = similar_subtitles_dir_non_mtv
            description_dir_sim = similar_description_dir_non_mtv
        else:
            print(f"Unknown label '{label}' for video ID '{similar_video_id}'. Skipping.")
            return None

        # Initialize default values
        thumbnail_description = None
        description = None
        subtitles = "Subtitles not available."
        explanation = None

        # Retrieve description
        description_path_sim = os.path.join(description_dir_sim, f"{similar_video_id}.txt")
        if os.path.exists(description_path_sim):
            with open(description_path_sim, 'r', encoding='utf-8') as desc_file:
                description = desc_file.read()
                print("d")

        # Retrieve subtitles
        subtitles_path_sim = os.path.join(subtitles_dir_sim, f"{similar_video_id}.txt")
        if os.path.exists(subtitles_path_sim):
            with open(subtitles_path_sim, 'r', encoding='utf-8') as subtitles_file:
                subtitles = subtitles_file.read()
                print("s")

        # Retrieve thumbnail description
        thumb_desc_path_sim = os.path.join(thumbnail_desc_dir, f"{similar_video_id}.txt")
        if os.path.exists(thumb_desc_path_sim):
            with open(thumb_desc_path_sim, 'r', encoding='utf-8') as thumb_desc_file:
                thumbnail_description = thumb_desc_file.read()
                print("t")

        # Retrieve explanation 
        explanation_path_sim = os.path.join(explanation_dir, f"{similar_video_id}.txt")
        if os.path.exists(explanation_path_sim):
            with open(explanation_path_sim, 'r', encoding='utf-8') as explanation_file:
                explanation = explanation_file.read()
                print("e")

        # Ensure all required fields are available
        if thumbnail_description and description and explanation:
            return {
                'thumbnail_description': thumbnail_description,
                'subtitles': subtitles,
                'description': description,
                'explanation': explanation
            }
        else:
            print(f"Missing required data for video ID '{similar_video_id}'.")
            return None

    except Exception as e:
        print(f"Error retrieving data for similar video ID '{similar_video_id}': {e}")
        return None


def truncate_text(text, max_words=200):
    """
    Truncates the input text to the specified maximum number of words.
    
    Parameters:
    - text (str): The text to truncate.
    - max_words (int): The maximum number of words allowed.
    
    Returns:
    - str: The truncated text.
    """
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    else:
        return text

# **Main Processing Loop**

for index, url in enumerate(df['url'], start=1):
    time.sleep(10)
    print(f"\nProcessing ({index}/{len(df)}): {url}")
    video_id = extract_video_id(url)
    
    if not video_id:
        print(f"Invalid YouTube URL: {url}. Skipping.")
        continue

    # Retrieve the Label from the dynamic CSV mapping
    # label was used for finding in which bucket is thumbnail stored, we can modify this by keeping all thumbnails in one bucket, no further use of label was done

    label = label_mapping.get(video_id)
    
    if not label:
        print(f"Label for video ID '{video_id}' not found in '{data_csv}'. Skipping.")
        continue

    # Define the Cloud Storage thumbnail path and bucket details based on Label
    if label == 'MTV':
        bucket_name = "uk-mtv-thumbs"
    elif label == 'Non MTV':
        bucket_name = "uk-non-mtv-thumbs"
    else:
        print(f"Unknown label '{label}' for video ID '{video_id}'. Skipping.")
        continue
    
    thumbnail_blob_name = f"{video_id}.jpg"

    # Check if the thumbnail exists in the Google Cloud Storage bucket
    thumbnail_exists = check_thumbnail_exists(bucket_name, thumbnail_blob_name)

    # Check if required description and subtitles files exist
    description_exists, subtitles_exists, description_path, subtitles_path = check_required_files(video_id)

    if description_exists and thumbnail_exists:
        try:
            # Retrieve description and subtitles
            video_description, video_subtitles = retrieve_files(video_id)

            # Define the Cloud Storage thumbnail URI
            thumbnail_uri = f"gs://{bucket_name}/{thumbnail_blob_name}"

            # **Find Similar Videos**
            similar_videos = get_top_similar_videos(
                video_description=video_description,
                video_id=video_id,
                data=data,  # Directly use the preloaded data DataFrame
                model=model,
                top_n=100
            )

            similar_examples = []

            # **Retrieve and Prepare Similar Videos' Data**
            for sim_video in similar_videos:
                sim_video_id = sim_video['Video ID']
                sim_label = sim_video['Label']
                sim_data = retrieve_similar_video_data(sim_video_id, sim_label)

                # Process only if all necessary fields are present
                if sim_data:
                    # **Categorization Based on Label**
                    categorization = "Misleading" if sim_label == 'MTV' else "Not Misleading"

                    similar_examples.append({
                        'thumbnail_description': sim_data['thumbnail_description'],
                        'subtitles': sim_data['subtitles'],
                        'description': sim_data['description'],
                        'categorization': categorization,
                        'explanation': sim_data['explanation']
                    })
                else:
                    print(f"Could not retrieve complete data for similar video ID '{sim_video_id}'.")

            # **Select 1 MTV and 1 Non MTV Example for the Prompt**
            selected_similar_examples = []
            mtv_selected = False
            non_mtv_selected = False
            for example in similar_examples:
                if not mtv_selected and example['categorization'] == "Misleading":
                    selected_similar_examples.append(example)
                    mtv_selected = True
                elif not non_mtv_selected and example['categorization'] == "Not Misleading":
                    selected_similar_examples.append(example)
                    non_mtv_selected = True
                if mtv_selected and non_mtv_selected:
                    break

            # **Prepare the Prompt with Selected Similar Examples**
            prompt_text = prepare_prompt(
                video_description=video_description,
                video_subtitles=video_subtitles,
                similar_examples=selected_similar_examples if selected_similar_examples else None
            )
            print(f"Prepared prompt for video ID: {video_id}")
            print("the prompt is")
            print(prompt_text)

            # **Send Prompt and Thumbnails to Gemini Model**
            print(f"Sending prompt for video ID {video_id} with thumbnail: {thumbnail_uri}")
            try:
                parts = [
                    Part.from_uri(thumbnail_uri, mime_type="image/jpeg"),  # Main Thumbnail
                ]
                # Add the text prompt
                parts.append(Part.from_text(prompt_text))

                response = vision_model.generate_content(parts)

                # Save the result
                output_file = os.path.join(results_dir, f"{video_id}.txt")
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(response.text)
                print(f"Saved response for video ID '{video_id}' to '{output_file}'.")

            except Exception as e:
                print(f"Failed to process video ID '{video_id}': {e}")

        except Exception as e:
            print(f"Failed to process video ID '{video_id}': {e}")
    else:
        print(f"Missing resources for video ID '{video_id}': Description Exists: {description_exists}, Thumbnails Exists: {thumbnail_exists}")
