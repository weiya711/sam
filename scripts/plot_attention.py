import scienceplots
import matplotlib.pyplot as plt
import numpy as np

print(plt.style.available)

# plt.style.use(['tableau-colorblind10'])
plt.style.use(['seaborn-talk'])

# create data
x = "Standard Attention"
y1 = np.array([3186834])
y2 = np.array([1198240])
y3 = np.array([2154964])
y4 = np.array([3188232])
count = [1,2]
labels=["Standard attention", "Fused attention"]


# plot bars in stack manner
fig, ax = plt.subplots()
# ax.bar(x, y1, width=0.5, color='r')
# ax.bar(x, y2, width=0.5, bottom=y1, color='b')
# ax.bar(x, y3, width=0.5, bottom=y1+y2, color='y')
# ax.bar("Fused Attention", y4, width=0.5, label="Fused")
#my_cmap = plt.cm.get_cmap('GnBu')
ax.bar(x, y1, width=0.5)
ax.bar(x, y2, width=0.5, bottom=y1)
ax.bar(x, y3, width=0.5, bottom=y1+y2)
ax.bar("Fused Attention", y4, width=0.5)
# plt.xlabel("Method")
plt.ylabel("# of cycles (in millions)")
plt.legend(["Tensor_mul", "Softmax", "Tensor_mul", "Fused Attention"])
plt.title("Standard Attention Vs Fused Attention")
plt.show()
ax.savefig("test.png", bbox_inches='tight', dpi=300)
