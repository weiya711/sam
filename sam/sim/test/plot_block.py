import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['2','4','8','16']
Ygirls = [4172,16116,63380,251412]
Xgirls = [3775,6419,11707,22283]
c = ["tab:blue","tab:orange", "tab:red", "indigo"]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Ygirls, width=0.4)
plt.bar(X_axis + 0.2, Xgirls, width=0.4)
  
plt.xticks(X_axis, X)
plt.xlabel("Block Size")
plt.ylabel("Cycles")
plt.title("Naive vs blocked representation for block sparse Q*$K^T$ ")
plt.show()
plt.savefig("test.png", bbox_inches='tight')
