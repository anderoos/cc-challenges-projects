from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# Get baseball and hockey categories
train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset='train', shuffle=True, random_state=108)

test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset='test', shuffle=True, random_state=108)

# Get features
# print(emails.target_names)

# Get target vectors
# 0 = rec.sport.baseball, 1 = rec.sport.hockey
# print(emails.target)
# print(emails.target_names)

# Instantiate CountVectorizer, train
counter = CountVectorizer()
counter.fit(train_emails.data + test_emails.data)

train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Instantiate MultinominalNB
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

# Print classifier accuracy
print(classifier.score(test_counts, test_emails.target))
