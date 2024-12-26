
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, DetectorFactory
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from joblib import Parallel, delayed
import nltk
import os

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Set langdetect seed for consistent results
DetectorFactory.seed = 0
tqdm.pandas() 

# Define input and output paths relative to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, '../data/raw/messi_comments.csv')
output_path = os.path.join(script_dir, '../data/processed/messi_comments_cleaned.csv')

# Load the dataset
def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

# Detect language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Clean text steps
def lowercase_text(text):
    return text.lower()

def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_special_chars(text):
    return re.sub(r'[^a-záéíóúñü\s]', '', text)

def remove_stopwords(text, lang):
    tokens = word_tokenize(text)
    if lang == 'en':
        stop_words = set(stopwords.words('english'))
    elif lang == 'es':
        stop_words = set(stopwords.words('spanish'))
    else:
        return text  # Skip stopword removal for unsupported languages
    return ' '.join([word for word in tokens if word not in stop_words])

def stem_text(text, lang):
    stemmer = SnowballStemmer(lang) if lang in ['spanish', 'english'] else None
    if stemmer:
        tokens = word_tokenize(text)
        return ' '.join([stemmer.stem(word) for word in tokens])
    return text

# Apply cleaning steps
def clean_text(row):
    raw_text, lang = row['raw_text'], row['detected_language']
    lowercased = lowercase_text(raw_text)
    no_urls = remove_urls(lowercased)
    no_special_chars = remove_special_chars(no_urls)
    no_stopwords = remove_stopwords(no_special_chars, lang)
    stemmed = stem_text(no_stopwords, lang)
    return pd.Series([lowercased, no_urls, no_special_chars, no_stopwords, stemmed])

# Parallel processing helper
def parallel_apply(df, func, columns):
    results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(func)(row) for _, row in tqdm(df.iterrows(), total=len(df)))
    return pd.DataFrame(results, columns=columns)

# Main cleaning process
def clean_dataset(input_path, output_path):
    df = load_data(input_path)

    # Filter for relevant type
    print("Filtering for type 'instagram_comment'...")
    df = df[df['type'] == 'instagram_comment']

    # Detect languages with progress bar
    print("Detecting languages...")
    df['detected_language'] = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(detect_language)(text) for text in tqdm(df['raw_text']))

    # Apply cleaning steps with progress bar
    print("Applying cleaning steps...")
    cleaning_columns = ['lowercased_text', 'text_no_urls', 'text_no_special_chars', 'text_no_stopwords', 'stemmed_text']
    cleaned_data = parallel_apply(df[['raw_text', 'detected_language']], clean_text, cleaning_columns)
    df[cleaning_columns] = cleaned_data

    # Save the cleaned dataset
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Data cleaning completed successfully.")

# Run cleaning process
if __name__ == '__main__':
    clean_dataset(input_path, output_path)
