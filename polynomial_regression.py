# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing our data set
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values # x should always be a matrix try to change it to [:,1] you will see the differnece
y=dataset.iloc[:,2].values



# Splitting the dataset into the Training set and Test set
# we won't splitt our data cause our information is only ten rows so small
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scalling 
# The library will takecare of it for us
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# you can scale your dummy variables or not based on the context and what you want to do
# with your data if you scalled it you will lose the knoweldge of the encding 
# but you may get better accuracy if you don't they will be already scaled for this model
# we will scale them
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting linear regression to the data
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

# Fitting polynomial regression to all the data
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


# Visualizing the linear regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial regression
X_grid=np.arange(min(X),max(X),0.1) # increment by 0.1 to get better curve
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluff (polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with linear regression
lin_reg.predict([[6.5]]) # 6.5 the level of our employee job 

# Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))