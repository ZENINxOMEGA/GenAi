import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 16,
        }
x = np.arange(1,11)
y1 = 3*x+5
y2 = 2*x+3
plt.plot(x,y1,label = 'Price',color = 'r',marker = '*',markersize = 15,markerfacecolor = 'b')
plt.plot(x,y2,label = 'Line',color = 'g',marker = 'o',linestyle=':')

plt.xlabel('X',fontdict=font)
plt.ylabel('f(X) = Y',fontdict = font)
plt.title("Line Plot",fontdict = font)
plt.legend()
# plt.grid()
# plt.savefig('LinePlot.jpg')
plt.show()