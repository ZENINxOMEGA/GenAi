import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = np.array([1,2,3,4,2,5,1,5,7])
y = x**2

plt.scatter(x,y,c = 'r')
plt.show()