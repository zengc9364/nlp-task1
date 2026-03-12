import pandas as pd
import numpy as np
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

new_text = "This colossal nation spans two different continents, making it a bridge between distinct cultural spheres. Because of its sheer size, traveling from its western borders to its eastern shores means passing through multiple time zones. The natural environment is notoriously harsh, featuring vast, freezing plains and endless stretches of dense, needle-leaf forests. Historically, it has transitioned through powerful imperial eras and a major twentieth-century ideological union, leaving a complex legacy. Today, its global influence is heavily sustained by the extraction and exportation of immense underground energy reserves, particularly fossil fuels, which lie hidden beneath its freezing terrain."
new_vector = vectorizer.transform([new_text])

similarity_scores = cosine_similarity(new_vector, tfidf_matrix)[0]
results = dict(zip(columns_names, similarity_scores))

print("similarity score:")
for country, score in results.items():
    print(f" {country}: {score:.4f}")

best_match = max(results, key=results.get)
print(f"\n The country closest to this new text is: {best_match}")

new_text_nonzero_indices = new_vector.nonzero()[1]

russia_index = columns_names.index("Russia")
russia_vector = tfidf_matrix[russia_index]
russia_nonzero_indices = russia_vector.nonzero()[1]

shared_indices = np.intersect1d(new_text_nonzero_indices, russia_nonzero_indices)

evidence_data = []
for idx in shared_indices:
    word = feature_names[idx]
    weight_in_new = new_vector[0, idx]
    weight_in_russia = russia_vector[0, idx]
    evidence_data.append({
        "Matched resonance words": word,
        "Weight in new text": round(weight_in_new, 4),
        "Weight in the original Russian text": round(weight_in_russia, 4)
    })

df_evidence = pd.DataFrame(evidence_data).sort_values(by="Weight in the original Russian text", ascending=False)
print(df_evidence.to_string(index=False))

best_match = max(results, key=results.get)
print(f"\n The country closest to this new text is: {best_match}")
