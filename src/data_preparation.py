#Functions for data cleaning, preprocessing, tokenization, and feature engineering

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from datasets import load_dataset
nltk.download('stopwords')

#Data Cleaning

def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)
    return text

arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

#Preprocessing

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

#Tokenization

def simple_word_tokenize(text):
    """
    Tokenize text into words / symbols with Arabic support.
    """
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)

def sentence_tokenize(text):
    """
    Split text into sentences using Arabic/English punctuation.
    """
    if not isinstance(text, str):
        return []
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def paragraph_tokenize(text):
    """
    Split text into paragraphs based on double newlines.
    """
    if not isinstance(text, str):
        return []
    paragraphs = re.split(r'\s*\n\s*\n\s*|\s*\r\n\s*\r\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

#Feature Engineering

full_df["tokens"] = full_df[clean_text_col].apply(lambda t: [tok for tok in simple_word_tokenize(t) if tok.strip()] if isinstance(t, str) else [])

full_df["words"] = full_df["tokens"].apply(lambda toks: [tok for tok in toks if re.search(r'\w', tok)])

full_df["sentences"] = full_df[original_text_col].apply(sentence_tokenize)

full_df["paragraphs"] = full_df[original_text_col].apply(paragraph_tokenize)

#Feature 16: Number of words with repeated letters
feature_name = f'{clean_text_col}_f016_words_with_repeated_letters'

def _words_with_repeated_letters(words):
    """Counts words containing at least one pair of adjacent identical letters."""
    if not words:
        return 0
    return sum(1 for w in words if re.search(r'(.)\1', w))

full_df[feature_name] = full_df["words"].apply(_words_with_repeated_letters)

#Feature 34: 34. Total number of sentences (S)
full_df['f034_Total_number_of_sentences_(S)'] = full_df["sentences"].apply(len)

#Feature 39: Average number of words/ S
full_df['f039_Average_words_per_sentence'] = full_df.apply(
    lambda row: len(row['words']) / row['f034_Total_number_of_sentences_(S)']
    if row['f034_Total_number_of_sentences_(S)'] > 0 else 0,
    axis=1)

#Feature 62: Number of imperfective
col = original_text_col
morph_features_col = f'{col}_morph_features'

if morph_features_col in full_df.columns:
    full_df[f'{col}_f062_num_imperfective'] = full_df[morph_features_col].apply(
        lambda d: sum(1 for perf in d.get('is_perfective', []) if not perf))
else:
    full_df[f'{col}_f062_num_imperfective'] = 0

#Feature 85: Sentence Length Variance: Variance in the number of words per sentence.
def sentence_length_variance(text):
    sentences = sentence_tokenize(text)
    if len(sentences) <= 1:
        return 0
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    return sum((l - mean_len) ** 2 for l in lengths) / len(lengths)

col = original_text_col
feature_name = f"{col}_f085_sentence_length_variance"

full_df[feature_name] = full_df[col].apply(sentence_length_variance)

#Feature 108: Politeness Score: Measures politeness.
polite_words = [
    "من فضلك", "شكراً", "لو سمحت", "عفواً",
    "من فضل حضرتك", "تكرماً", "فضلاً", "شاكراً لك",
    "مع الشكر", "أكون لك من الشاكرين", "متفضلاً",
    "أرجوك", "لطفاً", "إذا سمحت", "لو تكرمت",
    "شكراً جزيلاً", "جزاك الله خيراً", "بارك الله فيك",
    "شكراً لك", "أشكرك", "ممتن لك", "أكون ممتناً",
    "تفضّل", "من لطفك", "يسعدني", "أتمنى منك",
    "لو سمحتم", "لو تفضلتم", "بكل لطف", "كل الاحترام"
]
def politeness_score(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0

    try:
        count = sum(text.count(word) for word in polite_words)
        total_words = len(simple_word_tokenize(text))
        return (count / total_words) if total_words > 0 else 0.0
    except Exception:
        return 0.0

col = original_text_col
feature_name = f'{col}_f108_politeness_score'

full_df[feature_name] = full_df[col].apply(politeness_score)
