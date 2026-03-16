import re
import math

documents = [
    "Artificial Intelligence (AI) is a multidisciplinary field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. Core subfields include machine learning, deep learning, natural language processing, computer vision, and reinforcement learning.",
    "Computer Networks is a foundational field of information technology that focuses on the interconnection of computing devices to enable data exchange and resource sharing. It relies on standardized communication protocols, most notably the TCP/IP suite.",
    "Industrial Robotics is a specialized engineering field focused on the design, development, and deployment of automated robotic systems for manufacturing. Core components include robotic arms, servo motors, precision sensors, and controllers."
]

filenames = ["AI.txt", "Networks.txt", "Robotics.txt"]
TOP_K = 10 

class SimpleTfidf:
    def __init__(self):
        self.vocab = []
        self.idf = []

    def _tokenize(self, text):
        return re.findall(r'(?u)\b\w+\b', text.lower())

    def fit(self, docs):
        df = {}
        for doc in docs:
            for token in set(self._tokenize(doc)):
                df[token] = df.get(token, 0) + 1
        self.vocab = sorted(df.keys())
        self.idf = [math.log(len(docs)/df[token]) for token in self.vocab]

    def get_top_k(self, doc, k):
        tokens = self._tokenize(doc)
        tf = {}
        for token in tokens:
            if token in self.vocab:
                tf[token] = tf.get(token, 0) + 1
        scores = {t: tf[t] * self.idf[self.vocab.index(t)] for t in tf}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    tfidf = SimpleTfidf()
    tfidf.fit(documents)
    
    print(f"The first 10 words in the glossary: {tfidf.vocab[:10]}\n")
    for i, doc in enumerate(documents):
        top_k = tfidf.get_top_k(doc, TOP_K)
        print(f"doc '{filenames[i]}' Top-{TOP_K} Keywords:")
        for word, score in top_k:
            print(f"  {word}: {score:.4f}")
        print()
