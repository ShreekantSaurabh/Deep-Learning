#Artificial Neural Network

# Part 1 - Data Preprocessing
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#To avoid dummy variable trap, we have removed the first column of X
X = X[:, 1:]

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling (to make the features on same scale)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - making the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense          #builds layers of ANN

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#input_dim is for number of features we have. It will be used as input to the ANN. Mandatory parameter for first hidden layer.
#units is number of units in the layer. Here it is decided by (no. of nodes in input layer + no. of nodes in output layer)/2 = (11+1)/2 = 6
#kernel_initializer = 'uniform' initializes the weight uniformly

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#activation function for binary output is 'sigmoid', for higher no. of output is 'softmax'

# Compiling the ANN ie. adding Stochastic Gradient Descent  to the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer is algorith to optimize the weights during Back-propagation. 'adam' is one of the algo of SGD
#In adam (Adaptive Optimizer), initally alpha (learning rate) is high and gradually decreases with every epoch, where as the normal SGD alpha remains constant
#loss = 'binary_crossentropy' for two output of dependent variable and 'categorical_crossentropy' for more than 2 output
#metrics is criteria to improve the ANN model performance

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)
#batch_size is no. of observations/rows after which we are adjusting the weight
#One epoch = one Pass (forward & backward) through the ALgo or ANN
#One epoch has multiple iterations if batch size is defined.

#Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix to evaluate the prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
