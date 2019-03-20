import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#Note:- Data is artifically created in USA_Housing.csv file that's why it has decimal values in columns
df = pd.read_csv('USA_Housing.csv')
#df here stands for data frame
print(df.head())

print(df.info())

print(df.describe())

print(df.columns)

plt.show(sns.pairplot(df))

#distplot stands for distribution plot
plt.show(sns.distplot(df['Price']))

#df.corr() shows corelation between all the columns
print(df.corr())

plt.show(sns.heatmap(df.corr()))

plt.show(sns.heatmap(df.corr(),annot=True))

print(df.columns)

#x contain our features
x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

print(x)

#y is our taget variable.As we want to predict here house prices,so it will contain price column values
y=df['Price']
print(y)


#Now we split our data in train and test data
#We will use scikit learn to split data into train and test data as scikit learn comes with tain_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.4, random_state=101)
#test_size=0.4 here means 40% of your total dataset is radomly allocated as test data

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Fitting in the model
from sklearn.linear_model import LinearRegression

lm = LinearRegression() #creating object ( lm here )of type Linear Regression

lm.fit(x_train,y_train)  #Fitting and training the model on train data set

print(lm.intercept_)

print(lm.coef_) #This will print coefficient for each feature

print(x_train.columns)

cdf = pd.DataFrame(lm.coef_,x.columns,columns=['Coeff'])

print(cdf)


#Predictions section now

predictions = lm.predict(x_test)
print(predictions)

plt.show(plt.scatter(y_test,predictions))

plt.show(sns.distplot(y_test - predictions))


from sklearn import metrics

mae = metrics.mean_absolute_error(y_test,predictions)
print(mae)

mse = metrics.mean_squared_error(y_test,predictions)
print(mse)

#rmse stands for root mean squared error
rmse = np.sqrt(metrics.mean_squared_error(y_test,predictions))
print(rmse)
