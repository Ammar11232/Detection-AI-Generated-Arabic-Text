import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def plot_top_ngrams(texts, n=2, top_k=10, label="Dataset"):
    texts = texts.dropna().astype(str)

    vec = CountVectorizer(ngram_range=(n, n))
    bag = vec.fit_transform(texts)

    if bag.shape[1] == 0:
        print(f"No {n}-grams found for {label}.")
        return

    sum_words = bag.sum(axis=0)
    freqs = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]

    words, counts = zip(*freqs)

    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {top_k} {n}-grams for {label} abstracts")
    plt.tight_layout()
    plt.show()
