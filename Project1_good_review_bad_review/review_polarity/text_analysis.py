#importing libraries
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# dataset loading
reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target

#store in pickle format
with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)

#retrieve from pickle format
with open('X.pickle','rb') as f:
    X = pickle.load(f)

with open('y.pickle','rb') as f:
    y = pickle.load(f)

#creating the dataset
corpus = []
for i in range(0,len(X)):
    review = re.sub(r"\W"," ",str(X[i]))
    review = review.lower()
    review = re.sub(r"\s+[a-z]\s+"," ",review)
    review = re.sub(r"^[a-z]\s+"," ",review)
    review = re.sub(r"\s+"," ",review)
    corpus.append(review)

#bag of words model
vectorizer = CountVectorizer(max_features=2000,min_df=3,max_df=0.60,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

#tfidf model
transformer = TfidfTransformer()
X = transformer.fit_transform(X)

#or just Tfidf Vectorizer rather than both of the above
vectorizer = TfidfVectorizer(max_features=2000,min_df=3,max_df=0.60,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

#splitting the dataset
train_text,test_text,train_sent,test_sent = train_test_split(X,y,test_size=0.2,random_state=0)

#training on logistic regressor
classifier = LogisticRegression()
classifier.fit(train_text,train_sent)

sent_pred = classifier.predict(test_text)

cm = confusion_matrix(test_sent,sent_pred)
print("Accuracy = " + str((cm[0][0]+cm[1][1]) / 400) )

# converting the vectorizer and classifier to pickle format
with open('tfidf_vectorizer.pickle','wb') as f:
    pickle.dump(vectorizer,f)

with open('tfidf_classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

#loading vectorizer and classifier from pickle format
with open('tfidf_classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidf_vectorizer.pickle','rb') as f:
    vec = pickle.load(f)

#taking input from user
while(1):
    inp = input("Enter a string to try: ")
    inp = [inp]
    sample = vec.transform(inp).toarray()
    pred = clf.predict(sample)
    if pred == 0:
        print("\nNegative\n")
    else:
        print("\nPositive\n")
