# Criando um aplicativo da Web usando o Flask para apresentar um modelo de aprendizado de máquina

#Site fonte : https://iq.opengenus.org/web-app-ml-model-using-flask/


# Bibliotecas
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import pickle

# Conjunto de dados Iris
iris = pd.read_csv("Iris.csv")
print(iris.head())
iris.drop("Id", axis=1, inplace = True)
y = iris['Species']
iris.drop(columns='Species',inplace=True)
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# Modelo de regressão logística
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb')) # salva o modelo no disco usando o módulo pickle .