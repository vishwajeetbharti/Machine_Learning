import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model 
from word2number import w2n 

empl = pd.read_csv('hiring.csv')
empl.experience = empl.experience.fillna('zero')
empl.test_score = empl.test_score.fillna(empl.test_score.median())
for item in range(len(empl.experience)):
    empl.replace(empl.experience[item], f'{w2n.word_to_num(empl.experience[item])}', inplace=True)

reg = linear_model.LinearRegression()
reg.fit(empl[['experience','test_score','interview_score(out of 10)']].values,empl.salary.values)

plt.scatter(empl.experience,empl.salary,color='grey',marker='+')
plt.title('Salary of employ')
plt.xlabel('Experience')
plt.show()
val = reg.predict([[12,10,10]])
print(val)