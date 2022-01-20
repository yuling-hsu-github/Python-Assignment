import matplotlib.pyplot as plt
import numpy as np
a=[1,2,3,4,5,6,7,8]
data1=[1,4,9,16,25,36,49,64]
data2=[1,2,3,6,9,15,24,39]

plt.figure(figsize=(6,4),facecolor="lightgreen")
plt.title('Figure')
plt.ylim(0,70)
plt.xlim(0,8)
plt.plot(a,data2,"bo--",a,data1,'ro--')
plt.xlabel("x-Value",fontsize=16)
plt.ylabel("y-Value",fontsize=16)
plt.show()
