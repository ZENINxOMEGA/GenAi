import numpy as np
import matplotlib.pyplot as plt
def f(X):
    return 2*X**2-4*X+5

X = np.linspace(-3,5,100)
y = f(X)

plt.plot(X,y)
plt.show()

def derF(X):
    return 4*X-4
x = -2
alpha = 0.1
xv = [x]
yv = [f(x)]
for i in range(10):
    x = x - alpha*derF(x)
    xv.append(x)
    yv.append(f(x))
print(x)

plt.plot(X,y)
plt.plot(xv,yv,c = 'r')
plt.show()
