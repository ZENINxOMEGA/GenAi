import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('./GenAi\Linear_X_Train.csv').values.reshape((-1,))
y = pd.read_csv('./GenAi\Linear_Y_Train.csv').values.reshape((-1,))

plt.scatter(X,y)
plt.show()



w1 = 50
w0 = 1
lr = 0.1
yp = w1*X+w0

w1 = w1 - lr*np.mean((yp-y)*X)
w0 = w0 - lr*np.mean(yp-y)

plt.scatter(X,y)
plt.plot(X,yp,c='r')
plt.show()