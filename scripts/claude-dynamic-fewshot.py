import os
import pandas as pd
from anthropic import AnthropicVertex
import base64
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.",
    category=FutureWarning,
    module='transformers.tokenization_utils_base'
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set your location and project ID
LOCATION = "europe-west1"
PROJECT_ID = "your-project-id"
client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

# Load the CSV file containing YouTube URLs
csv_file = "/path/to/your/csv/"



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

# Directories for subtitles, thumbnails, descriptions, and results
'''For ease we will be evaluating NMTVs and NMTVs separately, uncomment the next four lines and comment the next 4 to evaluate NMTVs'''
# subtitles_dir = 'NMTV_Subtitles'
# thumbnails_dir = 'NMTV_Thumbnails'
# description_dir = 'Claude_NMTV_Video_To_Text'
# results_dir = 'Dynamic_NMTV_Claude'
subtitles_dir = 'MTV_Subtitles'
thumbnails_dir = 'MTV_Thumbnails'
description_dir = 'Claude_MTV_Video_To_Text'
results_dir = 'Dynamic_MTV_Claude'
os.makedirs(results_dir, exist_ok=True)

similar_thumbnails_dir_mtv = 'MTV_Thumbnails'
similar_thumbnails_dir_non_mtv = 'NMTV_Thumbnails'
similar_thumbnails_thumb_desc_dir_mtv = 'MTV_Thumbnail_Description'
similar_thumbnails_thumb_desc_dir_non_mtv = 'NMTV_Thumbnail_Description'
similar_video_explanations_dir_mtv = 'MTV_Explanations'
similar_video_explanations_dir_non_mtv = 'NMTV_Explanations'
similar_subtitles_dir_mtv = 'MTV_Subtitles'
similar_subtitles_dir_non_mtv = 'NMTV_Subtitles'
similar_description_dir_mtv = 'Claude_MTV_Video_To_Text'
similar_description_dir_non_mtv = 'Claude_NMTV_Video_To_Text'

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

# **Helper Function to Truncate Text to a Maximum Number of Words**
def truncate_text(text: str, max_words: int = 200) -> str:
    """
    Truncates the input text to the specified maximum number of words.
    If the text exceeds the limit, it appends an ellipsis.
    
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

# Function to extract the video ID from a YouTube URL
def extract_video_id(url: str) -> Optional[str]:
    if 'youtube.com/watch?v=' in url:
        return url.split('youtube.com/watch?v=')[-1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    else:
        return None

# Function to retrieve necessary files for a given video ID
def retrieve_files(video_id: str) -> (Optional[str], Optional[str], Optional[str]):
    subtitle_path = os.path.join(subtitles_dir, f"{video_id}.txt")
    description_path = os.path.join(description_dir, f"{video_id}.txt")
    thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}.jpg")
    
    if os.path.exists(description_path) and os.path.exists(thumbnail_path):
        # Read subtitle, use default message if not available
        if os.path.exists(subtitle_path):
            with open(subtitle_path, 'r', encoding='utf-8') as s_file:
                subtitles = s_file.read()
        else:
            subtitles = "Subtitles not available."
        
        with open(description_path, 'r', encoding='utf-8') as d_file:
            description = d_file.read()
        
        with open(thumbnail_path, 'rb') as thumb_file:
            thumbnail_base64 = base64.b64encode(thumb_file.read()).decode("utf-8")
        
        return subtitles, description, thumbnail_base64
    else:
        return None, None, None

def retrieve_similar_video_data(similar_video_id: str, label: str) -> Optional[Dict[str, str]]:
    """
    Retrieves the thumbnail description, subtitles, explanation, and description for a similar video based on its label.
    
    Parameters:
    - similar_video_id (str): The ID of the similar video.
    - label (str): The label of the similar video ('MTV' or 'Non MTV').
    
    Returns:
    - Dict[str, str]: A dictionary containing 'thumbnail_description', 'subtitles', 'description', and 'explanation'.
    """
    try:
        if label == 'MTV':
            # Paths for MTV videos
            thumbnail_desc_path = os.path.join(similar_thumbnails_thumb_desc_dir_mtv, f"{similar_video_id}.txt")
            explanation_path = os.path.join(similar_video_explanations_dir_mtv, f"{similar_video_id}.txt")
            subtitles_path = os.path.join(similar_subtitles_dir_mtv, f"{similar_video_id}.txt")
            description_path = os.path.join(similar_description_dir_mtv, f"{similar_video_id}.txt")
        elif label == 'Non MTV':
            # Paths for Non MTV videos
            thumbnail_desc_path = os.path.join(similar_thumbnails_thumb_desc_dir_non_mtv, f"{similar_video_id}.txt")
            explanation_path = os.path.join(similar_video_explanations_dir_non_mtv, f"{similar_video_id}.txt")
            subtitles_path = os.path.join(similar_subtitles_dir_non_mtv, f"{similar_video_id}.txt")
            description_path = os.path.join(similar_description_dir_non_mtv, f"{similar_video_id}.txt")
        else:
            print(f"Unknown label '{label}' for video ID '{similar_video_id}'. Skipping.")
            return None
        
        # Initialize dictionary to store the data
        video_data = {}

        # Read and store thumbnail description if available
        if os.path.exists(thumbnail_desc_path):
            with open(thumbnail_desc_path, 'r', encoding='utf-8') as d_file:
                video_data['thumbnail_description'] = d_file.read()

        # Read and store subtitles if available, else store "Subtitles not available."
        if os.path.exists(subtitles_path):
            with open(subtitles_path, 'r', encoding='utf-8') as s_file:
                video_data['subtitles'] = s_file.read()
        else:
            video_data['subtitles'] = "Subtitles not available."

        # Read and store description if available
        if os.path.exists(description_path):
            with open(description_path, 'r', encoding='utf-8') as d_file:
                video_data['description'] = d_file.read()

        # Read and store explanation if available
        if os.path.exists(explanation_path):
            with open(explanation_path, 'r', encoding='utf-8') as e_file:
                video_data['explanation'] = e_file.read()
        
        # Return the video data if any field is populated, otherwise None
        if video_data:
            return video_data
        else:
            print(f"No valid data found for similar video ID '{similar_video_id}'.")
            return None

    except Exception as e:
        print(f"Error retrieving data for similar video ID '{similar_video_id}': {e}")
        return None

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

# Function to prepare the prompt based on video description, subtitles, and thumbnails
def prepare_prompt(video_description: str, video_subtitles: str, similar_examples: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Prepares the prompt for the AI model, including any similar video examples.
    
    Parameters:
    - video_description (str): Description of the input video.
    - video_subtitles (str): Subtitles of the input video.
    - similar_examples (list of dict, optional): List containing similar video examples.
    
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
        for idx, example in enumerate(similar_examples, start=2):  
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

# Function to send the prompt and data to Claude
def send_to_claude(model_name: str, prompt: str, main_thumbnail_base64: str) -> Dict:
    """
    Sends the prepared prompt and thumbnail to the Claude model and retrieves the response.

    Parameters:
    - model_name (str): The name of the Claude model to use.
    - prompt (str): The formatted prompt to send to the model.
    - main_thumbnail_base64 (str): The base64-encoded thumbnail image.

    Returns:
    - Dict: The response from the Claude model.
    """
    prompt_messages = [
        {
            "role": "user",
            "content": [
                # **Main Video Thumbnail (Image 1)**
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": main_thumbnail_base64,
                    },
                },

                # **Text Prompt**
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
for index, url in enumerate(df['url'], start=1):
    time.sleep(10)
    print(f"\nProcessing ({index}/{len(df)}): {url}")
    video_id = extract_video_id(url)
    
    if video_id:
        subtitles, description, main_thumbnail_base64 = retrieve_files(video_id)
        
        if subtitles and description and main_thumbnail_base64:
            # **Find Similar Videos**
            similar_videos = get_top_similar_videos(
                video_description=description,
                video_id=video_id,
                data=data,
                model=model,
                top_n=100
            )
            
            # Initialize lists for MTV and Non MTV similar videos
            similar_mtv_list = []
            similar_non_mtv_list = []
            
            # Iterate through similar videos to find up to 10 MTV and 10 Non MTV
            for sim_video in similar_videos:
                if sim_video['Label'] == 'MTV' and len(similar_mtv_list) < 10:
                    similar_mtv_list.append(sim_video)
                elif sim_video['Label'] == 'Non MTV' and len(similar_non_mtv_list) < 10:
                    similar_non_mtv_list.append(sim_video)
                # Break early if both lists have 10 videos each
                if len(similar_mtv_list) == 10 and len(similar_non_mtv_list) == 10:
                    break
            
            # Combine the lists
            selected_similar_videos = similar_mtv_list + similar_non_mtv_list

            
            # Check if we have enough similar videos
            if len(selected_similar_videos) < 20:
                print(f"Warning: Only found {len(selected_similar_videos)} similar videos (MTV: {len(similar_mtv_list)}, Non MTV: {len(similar_non_mtv_list)}).")
            
            similar_examples = []
         
            # **Retrieve and Prepare Similar Videos' Data**
            for sim_video in selected_similar_videos:
                sim_video_id = sim_video['Video ID']
                label = sim_video['Label']
                sim_data = retrieve_similar_video_data(sim_video_id, label)

                # Check if sim_data exists and required fields (thumbnail_description, subtitles, description, and explanation) are available
                if sim_data and all(key in sim_data for key in ['thumbnail_description', 'subtitles', 'description', 'explanation']):
                    # **Corrected Categorization Based on Label**
                    # Assuming 'MTV' videos are "Misleading" and 'Non MTV' are "Not Misleading"
                    categorization = "Misleading" if label == 'MTV' else "Not Misleading"
                    
                    # Append valid data to the similar_examples list, including explanation
                    similar_examples.append({
                        'thumbnail_description': sim_data['thumbnail_description'],
                        'subtitles': sim_data['subtitles'],
                        'description': sim_data['description'],
                        'categorization': categorization,
                        'explanation': sim_data['explanation'],
                        'ID': sim_video_id
                    })
                    print(f"Appended similar example for video_id '{sim_video_id}'.")
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
                    print(f"MTV example ID:{example['ID']}")
                elif not non_mtv_selected and example['categorization'] == "Not Misleading":
                    selected_similar_examples.append(example)
                    non_mtv_selected = True
                    print(f"Non MTV example ID:{example['ID']}")
                if mtv_selected and non_mtv_selected:
                    break
            if not selected_similar_examples:
                print("No similar examples available to include in the prompt.")
            
            # **Prepare the Prompt with Selected Similar Examples**
            prompt_text = prepare_prompt(description, subtitles, similar_examples=selected_similar_examples if selected_similar_examples else None)
            # print("The prompt is:")
            # print(prompt_text)
            
            # **Send Prompt and Thumbnails to Claude for Each Model**
            for model_name in models:
                try:
                    print(f"Sending to Model: {model_name}")
                    response = send_to_claude(model_name, prompt_text, main_thumbnail_base64)
                    
                    # Accumulate text content from the response
                    text_content = ""
                    for block in response.content:
                        if block.type == 'text':
                            text_content += block.text + "\n"
                    
                    # Save the text content to a file
                    output_file = os.path.join(results_dir, f"{video_id}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text_content.strip())  # Save only text content
                    print(f"Saved response for video ID '{video_id}' to '{output_file}'")
                    
                except Exception as e:
                    print(f"Error with model '{model_name}': {e}")
        else:
            print(f"Invalid YouTube URL: {url}. Skipping.")


