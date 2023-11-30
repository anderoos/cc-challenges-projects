import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#import data
reviews = pd.read_csv('reviews.csv')
 
#print column names
# print(reviews.columns)

#print .info
# print(reviews.info())

#look at the counts of recommended
# print(reviews.recommended.value_counts())
 
#create binary dictionary
binary_dict = {True: 1, False: 0}
 
#transform column
reviews['recommended'] = reviews['recommended'].map(binary_dict)
 
#print your transformed column
# print(reviews.recommended.value_counts())

#look at the counts of rating
# print(reviews['rating'].value_counts())
 
#create dictionary
rating_dict = {
  "Loved it": 5,
  "Liked it": 4,
  "Was okay": 3,
  "Not great": 2,
  "Hated it": 1
}
 
#transform rating column
reviews['rating'] = reviews['rating'].map(rating_dict)
 
#print your transformed column values
# print(reviews['rating'].value_counts())

#get the number of categories in a feature
# print(reviews['department_name'].nunique())
 
#perform get_dummies
department_ohe = pd.get_dummies(reviews['department_name'])
 
#join the new columns back onto the original
reviews = reviews.join(department_ohe)

#print column names
# print(reviews)

#transform review_date to date-time data
# print(reviews['review_date'].dtype)
reviews['review_date'] = pd.to_datetime(reviews['review_date'])

#print review_date data type
# print(reviews['review_date'].dtype)

#get numerical columns

 
#reset index
reviews = reviews.set_index('clothing_id')

#instantiate standard scaler
scaler = StandardScaler()

#fit transform data
scaler.fit_transform(reviews)



