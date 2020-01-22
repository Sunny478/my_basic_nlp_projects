import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

url = input("Enter the URL of the wiki page: (try getting a lot of data) \n")
source = urllib.request.urlopen(url)

soup = bs.BeautifulSoup(source,'lxml')

text=""
for paragraph in soup.find_all('p'):
    text += paragraph.text

text = re.sub(r"\[[0-9]*\]"," ",text)
text = re.sub(r"[~!@#$%^&*_]<>?:{}\\[-]"," ",text)
text = text.lower()
text = re.sub(r"\W"," ",text)
text = re.sub(r"\d"," ",text)
text = re.sub(r"\s+"," ",text)

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

model = Word2Vec(sentences,min_count=1)

words = model.wv.vocab
for i in sorted(words.keys()):
    print(i,end="\n")
print("Length of words dictionary = "+str(len(words)))

while(1):

    pos = input("Enter the positive word: ")
    neg = input("Enter the negative word: ")
    for w in (model.wv.most_similar(positive=[pos],negative=[neg])):
        print(w[0])
    #print(model.wv.most_similar(positive=[pos],negative=[neg]))
