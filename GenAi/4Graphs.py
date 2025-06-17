import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = np.linspace(-2,2,1000)
sin = np.sin(2*np.pi*X)
cos = np.cos(2*np.pi*X)
tan = np.tan(2*np.pi*X)
cot = 1/np.tan(2*np.pi*X)

plt.figure(figsize=(10,12))

plt.subplot(2,2,1)
plt.plot(X,sin)
plt.grid()
plt.title("Sin Wave")

plt.subplot(2,2,2)
plt.plot(X,cos)
plt.grid()
plt.title("Cos Wave")

plt.subplot(2,2,3)
plt.plot(X,tan)
plt.grid()
plt.title("Tan Wave")

plt.subplot(2,2,4)
plt.plot(X,cot)
plt.grid()
plt.title("Cot Wave")

plt.suptitle('Trignometric Functions')
plt.subplots_adjust(hspace=0.3)
plt.show()