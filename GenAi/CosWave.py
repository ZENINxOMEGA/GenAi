import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = np.linspace(-2,2,1000)
y = np.cos(2*np.pi*X)

plt.figure(figsize=(10,2))
plt.plot(X,y)
plt.grid()
plt.title("Cos Wave")
plt.show()
