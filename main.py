import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

path1 = "pat1.txt"
path2 = "pat2.txt"

with open(path1, 'r') as file1:
    text1 = file1.read()

with open(path2, 'r') as file2:
    text2 = file2.read()

doc1 = nlp(text1.lower())
doc2 = nlp(text2.lower())
stop_words = set(nlp.Defaults.stop_words)
tokens1 = [token.lemma_.lower().strip() for token in doc1 if token.lemma_ and token.lemma_ not in stop_words]
tokens2 = [token.lemma_.lower().strip() for token in doc2 if token.lemma_ and token.lemma_ not in stop_words]

freq1 = {}
freq2 = {}
for token in tokens1:
    if token in freq1:
        freq1[token] += 1
    else:
        freq1[token] = 1
for token in tokens2:
    if token in freq2:
        freq2[token] += 1
    else:
        freq2[token] = 1

max_length = max(len(freq1), len(freq2))

vector1 = np.zeros(max_length)
vector2 = np.zeros(max_length)
for i, token in enumerate(freq1):
    vector1[i] = freq1[token]
for i, token in enumerate(freq2):
    vector2[i] = freq2[token]

similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

if similarity > 0.5:
    print(f"The texts are about the same topic. RES: {similarity}")
else:
    print(f"The texts are about different topics. RES: {similarity}")