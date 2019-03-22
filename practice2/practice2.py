from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB # classifier
from sklearn.svm import SVC # support vector machine - binary model
from random import shuffle # to shuffle our texts, probably they were combined in a wrong order
from sklearn.metrics import precision_recall_fscore_support

