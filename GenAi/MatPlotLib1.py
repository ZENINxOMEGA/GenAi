import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = np.linspace(-3,3,50)
y1 = 2*X+4
y2 = X**2 - 2

plt.plot(X,y1,label = 'y1')
plt.plot(X,y2,label = 'y2')

plt.grid()
plt.title('Some Functions')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axiz')
plt.legend()
plt.show()