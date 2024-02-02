from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
# import sklearn modules here:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# Setting up labels for your three friends
friends_labels = ['Emma'] * 154 + ['Matt'] * 141 + ['Tingfang'] * 166

# This mystery message
mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

def predictPostCard(string_seq, train_set, train_labels):
    # Vectorize String
    bow_vectorizer = CountVectorizer()
    friends_vector = bow_vectorizer.fit_transform(train_set)
    mystery_vector = bow_vectorizer.transform([string_seq])

    # NB Classifier
    friends_classifier = MultinomialNB()
    friends_classifier.fit(friends_vector, train_labels)
    predictions = friends_classifier.predict(mystery_vector)

    # Print classification results
    mystery_friend = predictions[0] if predictions[0] else "someone else"
    print("The postcard was from {}!".format(mystery_friend))
    probs = (friends_classifier.predict_proba(mystery_vector)*100)[0]
    formatted_probs = [f"{prob:.2f}%" for prob in probs]
    print(f'{formatted_probs}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictPostCard(mystery_postcard, friends_docs, friends_labels)