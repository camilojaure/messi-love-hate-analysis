
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
def load_data(input_path):
    return pd.read_csv(input_path)

# Cleaning steps
def lowercase_text(text):
    return text.lower()

def remove_urls(text):
    return re.sub(r'http\\S+|www\\.\\S+', '', text)

def remove_special_chars(text):
    return re.sub(r'[^a-záéíóúñü\\s]', '', text)

def remove_stopwords(text, lang='spanish'):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(lang))
    return ' '.join([word for word in tokens if word not in stop_words])

def stem_text(text, lang='spanish'):
    stemmer = SnowballStemmer(lang)
    tokens = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in tokens])

# Apply cleaning to dataset
def clean_dataset(input_path, output_path):
    df = load_data(input_path)
    if 'text' in df.columns:  # Replace 'text' with the correct column name for comments
        # Apply each cleaning step and save as new columns
        df['lowercased_text'] = df['text'].apply(lowercase_text)
        df['text_no_urls'] = df['lowercased_text'].apply(remove_urls)
        df['text_no_special_chars'] = df['text_no_urls'].apply(remove_special_chars)
        df['text_no_stopwords'] = df['text_no_special_chars'].apply(lambda x: remove_stopwords(x, lang='spanish'))
        df['stemmed_text'] = df['text_no_stopwords'].apply(lambda x: stem_text(x, lang='spanish'))
        
        # Save the cleaned dataset
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("Error: Expected column 'text' not found in dataset.")

# Run cleaning process
if __name__ == '__main__':
    input_path = '../data/raw/messi_comments.csv'
    output_path = '../data/processed/messi_comments_cleaned.csv'
    clean_dataset(input_path, output_path)
