import nltk
from nltk import pos_tag

sentence = "Westminster Abbey, formally titled the Collegiate Church of Saint Peter at Westminster, is a large, " \
           "mainly Gothic abbey church in the City of Westminster, London, England, just to the west of the Palace of " \
           "Westminster. It is one of the United Kingdom's most notable religious buildings and the traditional place " \
           "of coronation and burial site for English and, later, British monarchs. "

for sent in nltk.sent_tokenize(sentence):
    for chunk in nltk.ne_chunk(pos_tag(nltk.word_tokenize(sent))):
        if hasattr(chunk, 'label'):
            print (chunk.label(), ' '.join(c[0] for c in chunk))
