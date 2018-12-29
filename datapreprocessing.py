#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values


# Taking Care Of Missing Data i.e if there is Nan for any entity
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis= 0 )
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#Encoding Categorical Data i.e Quantitative into Qualitative
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(Y)



#Splitting Dataset into Training adn Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)