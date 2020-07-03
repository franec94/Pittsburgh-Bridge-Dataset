import numpy as np
import matplotlib.pyplot as plt

x1 = 10*np.random.random(100)
x2 = 10*np.random.exponential(0.5, 100)
x3 = 10*np.random.normal(0, 0.4, 100)
plt.boxplot ([x1, x2, x3])