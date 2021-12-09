#Shaghil's Part

#Mounting Our Drive On Cloab
from google.colab import drive
drive.mount('/content/drive')

#importing Libraries 
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Loading Training Data From Drive
train=panda.read_csv('/content/drive/MyDrive/smtp/train.csv')

#For Labels
y = train.Cover_Type

#For Features
X = train.drop('Cover_Type', axis=1)

#Splitting The Data Into 80%(For Training t_train) And 20%(For Testing t_test)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

#Faisal's Part
#Using No Smoothing For Data Fiting,Predicting,And Scoring Accuracy
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
NoMNB=clf.score(t_test,y_test)
print("The Accuracy Score Of NoMNB",NoMNB*100)


#Using Laplace Smoothing (Alpha=1) For Data Fiting,Predicting,And Scoring Accuracy
clf = MultinomialNB(alpha=1)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LPMNB=clf.score(t_test,y_test)
print("The Accuracy Score Of LPMNB",LPMNB*100)

#Using Lidstone  Smoothing (Alpha=0.5) For Data Fiting,Predicting,And Scoring Accuracy
clf = MultinomialNB(alpha=0.5)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LDMNB=clf.score(t_test,y_test)
print("The Accuracy Score Of LDMNB",LDMNB*100)


#Shaghil's Part

#Loading Training Data From Drive
test=panda.read_csv('/content/drive/MyDrive/smtp/test.csv')
test.head()


#Using No Smoothing For Data Fiting,Predicting,And Scoring Accuracy As it's Accuracy Was Good From Others
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
Cover_type=clf.predict(test)
print("The Predicted Values",Cover_type)

#Faisal's Part
#Exporting The Id And Cover_Type Columns Into Sample Csv
sample = test[['Id']].copy()
sample['Cover_Type'] = Cover_type
print(sample)


"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
sample.to_csv('sample.csv',index=False)
