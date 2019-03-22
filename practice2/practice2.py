from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # classifier
from sklearn.svm import SVC # support vector machine - binary model
from random import shuffle  # to shuffle our texts, probably they were combined in a wrong order
from sklearn.metrics import precision_recall_fscore_support, precision_score

"""READ DATA FROM FILES"""

f = open("training.txt", "r")

corpus = []

for i in f:
    corpus.append(i)

f.close()

sentences = list(enumerate(corpus))

shuffle(sentences)

train_data = sentences[:3500]
test_data = sentences[3500:]

train_corpus = []
train_category = []
mas = []

for s in train_data:
    mas = s[1].split("\t")
    train_corpus.append(mas[1])
    train_category.append(mas[0])

test_corpus = []
test_category = []

for s in test_data:
    mas = s[1].split("\t")
    test_corpus.append(mas[1])
    test_category.append(mas[0])

category = []
for i in train_category:
    category.append(i[0])
for i in test_category:
    category.append(i[0])

all_corpus = []
all_corpus.extend(train_corpus)
all_corpus.extend(test_corpus)

vectorizer = CountVectorizer(1)  # min_df=1
Matrix = vectorizer.fit_transform(all_corpus)  # word - frequency

# Split our matrix by test and train matrix
first_matrix = Matrix[:3500]
second_matrix = Matrix[3500:]

analyze = vectorizer.build_analyzer()

# Getting property (feature) vectors for classification
VF = vectorizer.get_feature_names()


# Train our model
clf = MultinomialNB().fit(first_matrix, [int(item) for item in train_category])

predicted = clf.predict(second_matrix)

vector_test = [int(item) for item in test_category]

print precision_recall_fscore_support(vector_test, predicted)
print precision_score(vector_test, predicted, 'micro')



#
# Matrix = vectorizer.
# tf_transformer = TfidfTransformer(use_idf=True).fit(Matrix) #use_idf=True
#
