import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model #Scipy provides low-level mathematical and scientific functions. scikit-learn provides high-level machine learning functions.


canada_income = pd.read_csv('canada_per_capita_income.csv')
df = pd.DataFrame(canada_income)

plt.scatter(df.year,df.per_capita_income_US,color='red',marker= '+')
plt.title('Capital Income')
plt.xlabel('Year')

reg = linear_model.LinearRegression()
reg.fit(df[['year']],df.per_capita_income_US)  #y = mX+b m is slope(or Gradient), b is y intercept
#value = reg.coef_ # 
# value1 = reg.intercept_
value2 = reg.predict([[2022]])
print(value2)
 
plt.plot(df.year,reg.predict(df[['year']]),color = 'blue')
plt.show()
