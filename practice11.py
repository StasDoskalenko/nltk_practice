import math
from collections import Counter


def compute_tfidf(corpus):
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i]/float(len(tf_text))
        return tf_text

    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list


def compute_idf(word, corpus):
    return math.log10(len(corpus) / sum([1.0 for i in corpus if word in i]))

corpus = [['pasta', 'la', 'vista', 'baby', 'la', 'vista'],
          ['hasta', 'siempre']]

# print compute_idf(corpus)

print compute_tfidf(corpus)

