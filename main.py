
#importing libraries
import pandas as panda
from sklearn.model_selection import train_test_split

#Getting the whole data set by mentioning its path of drive 
train=panda.read_csv('/content/drive/MyDrive/smtp/train.csv')
train.head()


#for labels
y = train.Cover_Type

#for features
X = train.drop('Cover_Type', axis=1)

#splitting the data set into 80% and 20%
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

#finally getting the splitting data
print("\nt_train:\n")
t_train.head()

print("\nt_test:\n")
t_test.head()
from sklearn.linear_model import LinearRegression
mnb = LinearRegression()
mnb.fit(t_train,t_train)
mnb.predict(t_test)
mnb.score(t_test,y_test)
