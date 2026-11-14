def chunk_by_words(text, max_words=250):
    words = str(text).split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
