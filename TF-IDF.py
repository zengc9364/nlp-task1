import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

raw_corpus = {
    "China": "China, officially the People's Republic of China, is an ancient civilization with over five thousand years of history and the world’s most populous country. Located in East Asia, it has diverse landscapes including plateaus, mountains, rivers and coastlines. China boasts rich cultural heritage such as the Great Wall, Forbidden City and traditional festivals. As a major global economy, it leads in manufacturing, trade, infrastructure and technological innovation. It follows socialism with Chinese characteristics and plays an increasingly important role in international affairs, global governance and regional cooperation. China values peace, development and win-win cooperation, striving to promote common prosperity and sustainable progress for all countries.",
    
    "Russia": "Russia, or the Russian Federation, is the largest country in the world by area, spanning Eastern Europe and northern Asia. It has a long history, profound literature, art and music traditions, and stunning natural scenery like forests, lakes and tundra. As a permanent member of the UN Security Council, Russia possesses strong national strength, advanced science and technology, and influential global status. It focuses on safeguarding national sovereignty, promoting economic development and deepening international partnerships. Russia values friendship and mutually beneficial cooperation with other nations, contributing to world peace, stability and balanced development.",
    
    "Pakistan": "Pakistan, officially the Islamic Republic of Pakistan, is a country in South Asia with a long history and unique Islamic culture. It has beautiful landscapes including mountains, plains and coastal areas, and is known for its warm and hospitable people. Pakistan enjoys close and all-weather friendly relations with China, deeply rooted in mutual trust and support. As a developing country, it is committed to economic growth, improving people’s livelihood and strengthening regional connectivity. Pakistan actively participates in international cooperation and advocates peace, justice and mutual respect in global affairs. It cherishes friendship with all nations and works to build a more peaceful and prosperous future."
}

documents = list(raw_corpus.values())
columns_names = list(raw_corpus.keys())

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(tfidf_matrix.T.toarray(), index=feature_names, columns=columns_names)

df_sorted_by_china = df_tfidf.sort_values(by="China", ascending=False)
df_sorted_by_Russia = df_tfidf.sort_values(by="Russia", ascending=False)
df_sorted_by_Pakistan = df_tfidf.sort_values(by="Pakistan", ascending=False)
pd.set_option('display.max_rows', None)

print("TF-IDF matrix after cleaning and filtering stop words")
print(df_tfidf.head(15))
print("\n====== China top5 word ======")
print(df_sorted_by_china.head(5))
print("\n====== Russia top5 word ======")
print(df_sorted_by_Russia.head(5))
print("\n====== Pakistan top5 word ======")
print(df_sorted_by_Pakistan.head(5))

new_text = "This powerful nation is situated in East Asia. It has a huge economy and extremely populous cities. Many tourists visit to see the Great Wall."
new_vector = vectorizer.transform([new_text])

similarity_scores = cosine_similarity(new_vector, tfidf_matrix)[0]

results = dict(zip(columns_names, similarity_scores))

print("similarity score:")
for country, score in results.items():
    print(f" {country}: {score:.4f}")

best_match = max(results, key=results.get)
print(f"\n The country closest to this new text is: {best_match}")
