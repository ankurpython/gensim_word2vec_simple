# Gensim Word2Vec Simple - Music Recommendation

A simple music recommendation system using Word2Vec to find similar songs based on playlist co-occurrence patterns.

## Overview

This project treats playlists as "sentences" and songs as "words" to learn song embeddings. Songs that appear together in playlists are considered similar, enabling music recommendations.

## Installation

```bash
pip install --upgrade gensim
pip install pandas numpy
```


### 2. Get Song Recommendations

```python
def print_recommendations(song_id):
    similar_songs = np.array(model.wv.most_similar(positive=str(song_id), topn=5))
    return songs_df.iloc[similar_songs]

# Get recommendations for song ID 2177
recommendations = print_recommendations(2177)
print(recommendations)
```

## Model Parameters

- `vector_size=100`: Dimension of song embeddings
- `window=20`: Context window size (songs considered together)
- `negative=50`: Number of negative samples
- `min_count=4`: Minimum song frequency threshold
- `workers=4`: Number of CPU cores to use

## How It Works

1. **Data Loading**: Fetches playlist data and song metadata from online dataset
2. **Preprocessing**: Filters playlists with multiple songs, skips metadata lines
3. **Training**: Uses Word2Vec to learn song embeddings from playlist patterns
4. **Recommendation**: Finds similar songs using cosine similarity in embedding space

