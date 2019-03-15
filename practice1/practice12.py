"""PERCEPTRON TAGGER and CHUNKING"""

from nltk import word_tokenize, ne_chunk
from nltk.tag import PerceptronTagger
from nltk.data import find

sentence = "Westminster Abbey, formally titled the Collegiate Church of Saint Peter at Westminster, is a large, " \
           "mainly Gothic abbey church in the City of Westminster, London, England, just to the west of the Palace of " \
           "Westminster. It is one of the United Kingdom's most notable religious buildings and the traditional place " \
           "of coronation and burial site for English and, later, British monarchs. "

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))
tagger = PerceptronTagger(False)
tagger.load(AP_MODEL_LOC)
pos_tag = tagger.tag
print ne_chunk(pos_tag(word_tokenize(sentence)))
