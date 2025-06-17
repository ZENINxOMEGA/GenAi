import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


X = np.linspace(-2,2,1000)
sin = np.sin(2*np.pi*X)
cos = np.cos(2*np.pi*X)

plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
plt.plot(X,sin)
plt.grid()
plt.title("Sin Wave")

plt.subplot(2,1,2)
plt.plot(X,cos)
plt.grid()
plt.title('Cos Wave')


plt.suptitle('Trignometric Functions')
plt.subplots_adjust(hspace=0.3)
plt.show()