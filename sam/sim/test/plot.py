import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['ijklm','ijkml','ijlkm','jimlk']
Ygirls = [538176,980056,545781,538176]
c = ["tab:blue", "tab:green", "tab:orange", "tab:red", "indigo"]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis, Ygirls, width=0.5, color=c)
  
plt.xticks(X_axis, X)
plt.xlabel("Reorder")
plt.ylabel("Cycles")
plt.title("Varying dataflow orders for Q*$K^T$ in attention computation")
plt.show()
plt.savefig("test.png", bbox_inches='tight')
