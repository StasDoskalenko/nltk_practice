"""PERCEPTRON TAGGER"""

from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "Mark and Jogn are working at Google."

from nltk.tag import PerceptronTagger
from nltk.data import find
PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))
tagger = PerceptronTagger(False)
tagger.load(AP_MODEL_LOC)
pos_tag = tagger.tag
print ne_chunk(pos_tag(word_tokenize(sentence)))
