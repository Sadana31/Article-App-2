from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df = pd.read_csv('articles.csv')
df = df[df['title'].notna()]

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['title'])

cosine_similarity2 = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['contentId'])

def get_recommendations(contentId):
    idx = indices[int(contentId)]
    similarityscores = list(enumerate(cosine_similarity2[idx]))
    similarityscores = sorted(similarityscores, key=lambda x: x[1], reverse=True)
    similarityscores = similarityscores[1:11]
    article_indices = [i[0] for i in similarityscores]
    return df[["url", "title", "text", "lang", "total_events"]].iloc[article_indices].values.tolist()