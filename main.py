import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
true['label'] = 1
fake['label'] = 0

# Concatenate dataframes
df = pd.concat([fake, true]).reset_index(drop = True)
df.shape
X = df.drop('label', axis=1)
y = df['label']

# Duplicate of the DataFrame for training & separate features & labels
df = df.dropna()
df2 = df.copy()
df2.head()
df2.reset_index(inplace=True)
df2.head()

#Pre-processing steps
#Stemming, lowercase and stopword
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import nltk
nltk.download('stopwords')

corpus = []
for i in range(0, len(df2)):
    review = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#######################################################################################
#Data exploration    
# Articles per subject
print(df.groupby(['subject'])['text'].count())
df.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

# Numbers of fake and real articles
print(df.groupby(['label'])['text'].count())
df.groupby(['label'])['text'].count().plot(kind="bar")
plt.show()


# Word cloud for fake news
from wordcloud import WordCloud

print(sys.executable)

df_fake = df[df["label"] == 0]
all_words = ' '.join([text for text in df_fake.text])

wordcloud = WordCloud(width= 800, height= 500,
                             max_font_size = 110,
                             collocations = False).generate(all_words)


plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Word cloud for real news
from wordcloud import WordCloud

df_real = df[df["label"] == 1]
all_words = ' '.join([text for text in df_real.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Most frequent words counter
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'purple')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()

# Most frequent words in fake news
counter(df[df["label"] == 0], "text", 20)

# Most frequent words in real news
counter(df[df["label"] == 1], "text", 20)

##############################################################################################
#Apply feature extraction
#TFidfVect for Data
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer()
X = tfidf_v.fit_transform(corpus).toarray()
y = df2['label']


# Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#######################################################################################
#Make Confusion matrix
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
#############################################################################

#LogisticRegression
#Applying TF-IDF
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
model = LogisticRegression()

# Fitting the model
model.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
score = metrics.accuracy_score(y_test, prediction)
print("accuracy:   %0.3f" % score)
print(classification_report(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes= ['FAKE', 'REAL'])
plt.show()

#############################################################################

#DecisionTreeClassifier
#Applying TF-IDF
import itertools
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# Fitting the model
model.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
score = metrics.accuracy_score(y_test, prediction)
print("accuracy:   %0.3f" % score)
print(classification_report(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

############################################################################

#SVM
#Applying TF-IDF
import itertools
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
classifier = LinearSVC()


# Fitting the model
classifier.fit(X_train, y_train)

# Accuracy
prediction = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, prediction)
print("accuracy:   %0.3f" % score)
print(classification_report(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes= ['FAKE', 'REAL'])
plt.show()

#############################################################################

#Passive Aggressive Classifier
# Applying TF-IDF
import itertools
from sklearn.metrics import classification_report
from sklearn.linear_model import PassiveAggressiveClassifier
model = PassiveAggressiveClassifier()


# Fitting the model
model.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
score = metrics.accuracy_score(y_test, prediction)
print("accuracy:   %0.3f" % score)
print(classification_report(y_test, prediction))


cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes= ['FAKE', 'REAL'])
plt.show()

##############################################################################
#pickle (save) the model and vectorizer
val = tfidf_v.transform([review]).toarray()
classifier.predict(val)

#pickle and load model
import pickle
pickle.dump(classifier, open('model2.pkl', 'wb'))
pickle.dump(tfidf_v, open('tfidfvect2.pkl', 'wb'))

# Load model and tfidf
joblib_model = pickle.load(open('model2.pkl', 'rb'))
joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))
val_pkl = joblib_vect.transform([review]).toarray()
joblib_model.predict(val_pkl)
