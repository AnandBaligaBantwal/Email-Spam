import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

spam_df = pd.read_csv("emails.csv")

spam_df.head()

spam_df.shape

spam_df.duplicated().sum()

spam_df[spam_df.duplicated()]

spam_df.drop_duplicates(inplace=True)

spam_df.isnull().sum()

spam_df["spam"].value_counts()

# Count Vectorizer on the text column

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorized_input_data = vectorizer.fit_transform(spam_df["text"])

vectorized_input_data.shape

vectorizer.get_feature_names_out()[8000:8100]

# Machine Learning Process
X = vectorized_input_data
y = spam_df["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Multinomial Naive Bayes Algorithm to the data

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

roc_auc_score(y_test, y_pred)
