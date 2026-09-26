import re
import pandas as pd

LEGAL_SUFFIXES = [
    r'\bltd\b', r'\blimited\b', r'\bpvt\b', r'\bprivate\b',
    r'\binc\b', r'\bincorporated\b', r'\bcorp\b', r'\bcorporation\b',
    r'\bllc\b', r'\bco\b', r'\bcompany\b'
]
LEGAL_REGEX = re.compile('|'.join(LEGAL_SUFFIXES), flags=re.IGNORECASE)
NON_ASCII_REGEX = re.compile(r'[^\x00-\x7F]+')
URL_REGEX = re.compile(r'http[s]?://\S+|www\.\S+')
PUNCTUATION_REGEX = re.compile(r'[^\w\s&]')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = URL_REGEX.sub('', text)
    # Remove non-ASCII
    text = NON_ASCII_REGEX.sub(' ', text)
    # Remove punctuation (keep &)
    text = PUNCTUATION_REGEX.sub(' ', text)
    # Extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df):
    """
    Applies text cleaning and extracts useful heuristic components.
    """
    df_clean = df.copy()
    
    # Basic cleaning
    df_clean['name_clean'] = df_clean['business_name'].apply(clean_text)
    df_clean['address_clean'] = df_clean['business_address'].apply(clean_text)
    
    # Strip legal entities
    df_clean['name_core'] = df_clean['name_clean'].apply(lambda x: LEGAL_REGEX.sub('', x).strip())
    df_clean['name_core'] = df_clean['name_core'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # Extract numbers from address (for numerical conflict feature)
    # Returns a sorted space-separated string of numbers found in the address
    df_clean['address_numbers'] = df_clean['address_clean'].apply(
        lambda x: " ".join(sorted(re.findall(r'\b\d+\b', x)))
    )
    
    # For embedding blocking, we concatenate core name and clean address
    df_clean['blocking_text'] = df_clean['name_core'] + " " + df_clean['address_clean']
    
    return df_clean
