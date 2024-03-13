import spacy
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")

# txt dosyasındaki cümleleri okuma ve spaCy ile işleme
with open("islenmiscorpus.txt", "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file]

# Boş bir liste oluşturdum
tokenized_sentences = []

# Cümleleri spaCy ile işledim
for sentence in tqdm(sentences, desc="Processing Sentences"):
    doc = nlp(sentence)
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    tokenized_sentences.append(tokens)

# Word2Vec modelini oluşturdum
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Derlemin içerisinde en çok geçen 20 kelimeyi bulma
word_freq = Counter([word for sentence in tokenized_sentences for word in sentence])
top_20_words = [word for word, _ in word_freq.most_common(20)]
# Her kelimenin frekansını gösteren kod
word_frequencies = {word: word_freq[word] for word in top_20_words}
print("Her kelime ve frekansı", word_frequencies)

# En çok geçen 20 kelimenin her biri için en benzer 5 kelimeyi bulma
similar_words_dict = {}
for word in top_20_words:
    similar_words = model.wv.most_similar(word, topn=5)
    similar_words_dict[word] = [similar_word for similar_word, _ in similar_words]
# 5 farklı cümle seçin
selected_sentences = [tokenized_sentences[i] for i in [500, 1000, 1500, 2000, 2500]]

# Her bir seçili cümle için en benzer 3 cümleyi bulma
similar_sentences_dict = {}
for i, selected_sentence in enumerate(selected_sentences):
    similar_sentences = []
    if selected_sentence:  
        for j, sentence in enumerate(tokenized_sentences):
            if i != j and sentence and selected_sentence: 
                similarity = model.wv.n_similarity(selected_sentence, sentence)
                similar_sentences.append((j, similarity))
        similar_sentences.sort(key=lambda x: x[1], reverse=True)
        similar_sentences_dict[i] = similar_sentences[:3]
    else:
        print(f"Warning: Selected sentence {i} is empty.")


# Sonuçları yazdırma
print("En çok geçen 20 kelime:", top_20_words)
print("\nHer bir kelimenin en benzer 5 kelimesi:")
for word, similar_words in similar_words_dict.items():
    print(f"{word}: {similar_words}")
# Seçili cümleleri ve benzer cümleleri yazdırma
print("\nSeçili 5 farklı cümle:")
for i, selected_sentence in enumerate(selected_sentences):
    print(f"Seçili {i + 1}. cümle: {' '.join(selected_sentence)}")
    print("En benzer 3 cümle:")
    for j, similarity in similar_sentences_dict[i]:
        print(f"  - Derlemdeki {j + 1}. cümle (Benzerlik: {similarity:.5f}): {' '.join(tokenized_sentences[j])}")