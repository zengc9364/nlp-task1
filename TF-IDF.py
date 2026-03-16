import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

raw_corpus = {
    "Artificial_Intelligence": "Artificial Intelligence (AI) is a multidisciplinary field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. Core subfields include machine learning, deep learning, natural language processing, computer vision, and reinforcement learning. Modern AI is driven by large language models, neural networks with billions of parameters, which power generative AI tools for content creation, code generation, and conversational interfaces. AI is widely applied across industries: healthcare for medical image analysis, finance for fraud detection, automotive for autonomous driving, and education for personalized learning. It continues to advance rapidly, pushing the boundaries of what intelligent systems can achieve, while also sparking discussions about ethics, bias mitigation, and responsible AI deployment.",
    
    "Computer_Networks": "Computer Networks is a foundational field of information technology that focuses on the interconnection of computing devices to enable data exchange and resource sharing across geographic distances. It relies on standardized communication protocols, most notably the TCP/IP suite, which governs how data is packetized, routed, and delivered between endpoints. Key concepts include network topology, routing algorithms, bandwidth management, network security, and cloud networking. Modern networks support critical infrastructure: from local area networks (LANs) in offices and homes, to wide area networks (WANs) connecting global data centers, to 5G and satellite networks enabling mobile and remote connectivity. Network professionals focus on optimizing performance, ensuring reliability, and defending against cyber threats like hacking, malware, and DDoS attacks.",
    
    "Industrial_Robotics": "Industrial Robotics is a specialized engineering field focused on the design, development, and deployment of automated robotic systems for manufacturing and industrial production. Core components of industrial robots include robotic arms, servo motors, precision sensors, controllers, and end-of-arm tooling tailored for specific tasks. These systems are engineered to perform repetitive, high-precision, or hazardous operations with consistent accuracy, such as welding, assembly, material handling, painting, and quality inspection. A cornerstone of Industry 4.0, industrial robots integrate with IoT sensors, machine learning, and digital twin technology to create smart, flexible production lines. They are widely used in automotive manufacturing, electronics assembly, food processing, and logistics, helping businesses improve efficiency, reduce labor costs, and enhance workplace safety."
}

documents = list(raw_corpus.values())
columns_names = list(raw_corpus.keys())

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(tfidf_matrix.T.toarray(), index=feature_names, columns=columns_names)
df_sorted_by_ai = df_tfidf.sort_values(by="Artificial_Intelligence", ascending=False)
df_sorted_by_networks = df_tfidf.sort_values(by="Computer_Networks", ascending=False)
df_sorted_by_robotics = df_tfidf.sort_values(by="Industrial_Robotics", ascending=False)
pd.set_option('display.max_rows', None)


new_text = "This cutting-edge technology uses layered neural networks to process and generate human-like text, images, and code. It powers chatbots that can hold natural conversations, tools that help developers write and debug software, and systems that create original art and marketing content. At its core are large models trained on massive datasets, using machine learning techniques to recognize patterns and make predictions. It is transforming industries from content creation to customer service, with ongoing research focused on improving accuracy, reducing harmful outputs, and making systems more accessible to businesses of all sizes."
new_vector = vectorizer.transform([new_text])

similarity_scores = cosine_similarity(new_vector, tfidf_matrix)[0]
results = dict(zip(columns_names, similarity_scores))

print("similarity score:")
for domain, score in results.items():
    print(f" {domain}: {score:.4f}")

best_match = max(results, key=results.get)
print(f"\n The domain closest to this new text is: {best_match}")

new_text_nonzero_indices = new_vector.nonzero()[1]
best_match_index = columns_names.index(best_match)
best_match_vector = tfidf_matrix[best_match_index]
best_match_nonzero_indices = best_match_vector.nonzero()[1]

shared_indices = np.intersect1d(new_text_nonzero_indices, best_match_nonzero_indices)

evidence_data = []
for idx in shared_indices:
    word = feature_names[idx]
    weight_in_new = new_vector[0, idx]
    weight_in_original = best_match_vector[0, idx]
    evidence_data.append({
        "Matched resonance words": word,
        "Weight in new text": round(weight_in_new, 4),
        f"Weight in the original {best_match} text": round(weight_in_original, 4)
    })

df_evidence = pd.DataFrame(evidence_data).sort_values(by=f"Weight in the original {best_match} text", ascending=False)
print(df_evidence.to_string(index=False))

best_match = max(results, key=results.get)
print(f"\n The domain closest to this new text is: {best_match}")
