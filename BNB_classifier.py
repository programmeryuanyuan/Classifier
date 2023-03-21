import os
os.environ["SKLEARN_SITE_JOBLIB"] = "TRUE"

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
""" import plot """

input_file = sys.argv[1]

train_data_file = pd.read_csv(input_file, sep = '\t', header=None, names=['instance_number','ranks','reviews'], nrows=2000)
test_data_file = pd.read_csv(input_file, sep = '\t', header=None, names=['instance_number','ranks','reviews'], skiprows=2000, nrows=500)


testing_id = np.array(test_data_file['instance_number'])
training_sentence, training_y = np.array(train_data_file['reviews']), np.array(train_data_file['ranks'])
testing_sentence, testing_y = np.array(test_data_file['reviews']), np.array(test_data_file['ranks'])

""" labels = ['1','2','3','4','5'] """
# Using sentiments
bins = [0,4,5,6]
labels = ['negative','neutral','positive']

training_y = pd.cut(
    training_y,
    bins,
    right=False, #left close, right open
    labels=labels
)
testing_y = pd.cut(
    testing_y,
    bins,
    right=False, #left close, right open
    labels=labels
)
# model
def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    for i in range(0,len(predicted_y)):
        print(testing_id[i], predicted_y[i])
    #print(classification_report(test_review, predicted_y,zero_division=0))

def data_preprocessing(sentence):
    replce_format = re.compile(r'\.{3}|~{1}|-{2}')
    remove_format = re.compile(r'<.*?>')
    # same, use regular expression to express the invalid char
    result = []
    for i in sentence:
        # replace ... to space
        replace_word_format = re.sub(replce_format, ' ', i)
        # remove format
        replace_word_format = re.sub(remove_format, '', i)
        """ # change upper letter to lower letter
        replace_word_format = re.sub(r'\w+', lambda a: a.group().lower(), i) """
        #print(replace_word_format)
        result.append(replace_word_format)
    return result


valid_train_sentence = data_preprocessing(training_sentence)
valid_test_sentence = data_preprocessing(testing_sentence)

token = r'[#@_$%\w\d]{2,}'
#max_features=1000,
count = CountVectorizer(max_features=1000, token_pattern=token, lowercase=False)

X_train_bag_of_words = count.fit_transform(valid_train_sentence)
X_test_bag_of_words = count.transform(valid_test_sentence)

clf = BernoulliNB()
model = clf.fit(X_train_bag_of_words, training_y)
predict_y = model.predict(X_test_bag_of_words)

# classification report
#predict_and_test(model, X_test_bag_of_words)

#python BNB_classifier.py reviews.tsv outputBNB.txt
output_file = sys.argv[2]
data = open(output_file,'w+')
# print the valid output
#print(predict_and_test(model, X_test_bag_of_words),file=data)
for i in range(len(testing_sentence)):
    print(f'{testing_id[i]} {predict_y[i]}',file=data)
data.close()

my_report = classification_report(testing_y, predict_y, zero_division=0)
print(my_report)

""" # Plot classifier report
plot.plot_classification_report(testing_y, predict_y, ml_name='BNB',
                                  classes=labels,
                                  title='Classification report for BNB_classifier') """