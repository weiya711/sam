# importing package
import matplotlib.pyplot as plt
import numpy as np
 
# create data
x = "Standard Attention"
y1 = np.array([7902])
y2 = np.array([1106])
y3 = np.array([5655])
y4 = np.array([7938])
count = [1,2]
labels=["Standard attention", "Fused attention"]

 
# plot bars in stack manner
fig, ax = plt.subplots()
# ax.bar(x, y1, width=0.5, color='r')
# ax.bar(x, y2, width=0.5, bottom=y1, color='b')
# ax.bar(x, y3, width=0.5, bottom=y1+y2, color='y')
# ax.bar("Fused Attention", y4, width=0.5, label="Fused")
my_cmap = plt.cm.get_cmap('GnBu')
ax.bar(x, y1, width=0.5)
ax.bar(x, y2, width=0.5, bottom=y1)
ax.bar(x, y3, width=0.5, bottom=y1+y2)
ax.bar("Fused Attention", y4, width=0.5)
# plt.xlabel("Method")
plt.ylabel("# of cycles")
plt.legend(["Tensor_mul", "Softmax", "Tensor_mul", "Fused Attention"])
plt.title("Standard Attention Vs Fused Attention")
plt.show()
ax.savefig("test.png", bbox_inches='tight')
