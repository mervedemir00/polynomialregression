import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#data load
datas= pd.read_csv('maaslar.csv')

x=datas.iloc[:,1:2]

y=datas.iloc[:,2:]

#numpy array conversion
X=x.values
Y=y.values

#linear regression
from sklearn.linear_model import LinearRegression 

lin_reg= LinearRegression()

lin_reg.fit(X,Y)

plt.scatter(X, Y)
plt.plot(X,lin_reg.predict(X))
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=4)

x_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(x,y)

plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))

plt.show()

