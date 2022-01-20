
'''excercise10'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
height=np.array([147,163,159,155,163,158,172,161,153,161])
weight=np.array([41,60,47,53,48,55,58.5,49,46,52.5])
x=pd.DataFrame(height,columns=["Height"])
y=pd.DataFrame(weight,columns=["Weight"])


lm=LinearRegression()
lm.fit(x,y)
print("線性回歸",lm.coef_)
print("截距",lm.intercept_)

new_weight=pd.DataFrame(np.array([155,165,180]),columns=["Height"])
predicted_weight=lm.predict(new_weight)
print(predicted_weight.round(1))


b=pd.DataFrame(height,columns=["Height"])
a=pd.DataFrame(weight,columns=["Weight"])
lm=LinearRegression()
lm.fit(a,b)
print("線性回歸",lm.coef_)
print("截距",lm.intercept_)

new_height=pd.DataFrame(np.array([55,65,70]),columns=["Weight"])
predict_height=lm.predict(new_height)
print(predict_height.round(0))


plt.scatter(weight,height)
regression_height=lm.predict(a)
plt.plot(weight,regression_height,color="black")
plt.plot(new_height,predict_height,color="red")
plt.show()

"""
#11:Diabetes
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
diabetes=datasets.load_diabetes()
x=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)

target=pd.DataFrame(diabetes.target,columns=["MEDV"])
print(target.head())
y=target["MEDV"]

lm=LinearRegression()
lm.fit(x,y)
# coef=pd.DataFrame(diabetes.feature_names,columns=["features"])
# coef["estimateCoefficients"]=lm.coef_#直接添加新的列在coef裡
# print(coef)
predicted_result=lm.predict(x)
plt.scatter(y,predicted_result)
plt.xlabel("name")
plt.ylabel("predicted results")
plt.title("Diabetes")
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
diabetes=datasets.load_diabetes()
x=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
target=pd.DataFrame(diabetes.target,columns=["MEDV"])
print(target.head())
y=target["MEDV"]
x1=x.drop(columns=["s1","s2","s3","s4","s5","s6"])#再見模之前就要把資料給整理好要部模型跑不出話
lm=LinearRegression()
lm.fit(x1,y)
predicted_result=lm.predict(x1)
print(predicted_result)
plt.scatter(y,predicted_result)
plt.xlabel("name")
plt.ylabel("predicted results")
plt.title("Diabetes")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
diabetes=datasets.load_diabetes()
x=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
target=pd.DataFrame(diabetes.target,columns=["MEDV"])
y=target["MEDV"]
lm=LinearRegression()
lm.fit(x,y)
predicted_x=lm.predict(x)
print(predicted_x)
MSE=np.mean((y-predicted_x)**2)
print("MSE:",MSE)
print("R-square:",lm.score(x,y))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
diabetes=datasets.load_diabetes()
x=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
target=pd.DataFrame(diabetes.target,columns=["MEDV"])
y=target["MEDV"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25, random_state=100)#亂數固定 讓測試資料及都是用同組數值以免數字一直跑
lm=LinearRegression()
lm.fit(xTrain,yTrain)

pred_train=lm.predict(xTrain)
pred_test=lm.predict(xTest)
MSE_Train=np.mean((yTrain-pred_train)**2)
MSE_test=np.mean((yTest-pred_test)**2)
print("MSE Train:",MSE_Train)
print("MSE Test:",MSE_test)
print("R-square Train:",lm.score(xTrain, yTrain))

plt.scatter(yTest,pred_test)
plt.xlabel("number")
plt.ylabel("predicted number")
plt.title("Number V.S. Predicted Number ")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
diabetes=datasets.load_diabetes()
x=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
target=pd.DataFrame(diabetes.target,columns=["MEDV"])
y=target["MEDV"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=100)#亂數固定 讓測試資料及都是用同組數值以免數字一直跑
lm=LinearRegression()
lm.fit(xTrain,yTrain)

pred_train=lm.predict(xTrain)
pred_test=lm.predict(xTest)
MSE_Train=np.mean((yTrain-pred_train)**2)
MSE_test=np.mean((yTest-pred_test)**2)
print("MSE Train:",MSE_Train)
print("R-square Train:",lm.score(xTrain, yTrain))

'''Breast Cancer'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
breast_cancer=datasets.load_breast_cancer()
x=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
target=pd.DataFrame(breast_cancer.target,columns=["Index"])#只是為了需要一個欄位名稱所以增加的欄位名稱
y=target["Index"]
