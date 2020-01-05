import os
import io
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

cd = os.getcwd()
print(cd + '/DataScience_CourseMaterial/emails/spam')


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False  # Added for skipping header
            lines = []
            f = io.open(path, 'r', encoding='latin1')

            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True

        f.close()
        message = '\n'.join(lines)
        yield path, message
        # using yield insted of return, returns a generator (object)


def dataFrameFromDir(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)  # returning as a DataFrame object


data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDir(cd +
                    '/DataScience_CourseMaterial/emails/spam', 'spam'))
data = data.append(dataFrameFromDir(cd +
                    '/DataScience_CourseMaterial/emails/ham', 'ham'))

print(data.head())

# Splits the email up to single words and counts them
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

# Classifies the email based on counts
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)


# Testing the classifier
examples = ['Free Viagra now!!!', 'Hi Bob, how about a game of golf tomorrow?']
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
