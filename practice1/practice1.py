"""
PREPOSSESSING
1) Cleaning
2) Annotation
3) Normalization - different city namings to a single form
4) POS tagger
"""

import nltk

text = "This approach still has many problems but we will mova on define a function to perform stemming. " \
       "And apply it to a whole text "

sentences = nltk.tokenize.sent_tokenize(text)
# print sentences
# print sentences[0]

sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
print sentences[0]

"""
Stemmer < stemmer faster than lemmatizer
"""
# stemmer = nltk.stem.SnowballStemmer("english")
# print [stemmer.stem(word) for word in sentences[0]]

"""Lancaster"""

lancaster = nltk.LancasterStemmer()
print [lancaster.stem(t) for t in sentences[0]]

"""WordNet lemmatization"""
wnl = nltk.WordNetLemmatizer()
print "WordNet"
print [wnl.lemmatize(t) for t in sentences[0]]

"""Stop words"""
# print nltk.corpus.stopwords.words("english")


def content_fraction(text_param):
    from nltk.corpus import stopwords
    stopwords = stopwords.words("english")
    content = [w for w in text_param if w.lower() not in stopwords]
    return len(content) / len(text_param)


# print content_fraction(nltk.corpus.reuters.words())  # 0 ???

"""Start POS tagger"""

tagged_sentence = nltk.pos_tag(sentences[0])
print tagged_sentence

tree = nltk.ne_chunk(tagged_sentence)
tree.draw()

"""End POS tagger"""
