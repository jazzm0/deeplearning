import matplotlib.pyplot as plt
import numpy as np

p = np.linspace(0, 7, 20)

# plt.plot([1, 2, 4, 8, 16], [1, 0, 1, 0, 1], 'bs-')
# plt.show()

plt.plot(p, np.sin(p), 'rs')
plt.show()
print()
