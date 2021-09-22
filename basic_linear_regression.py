import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
from reverse_autodiff import Node as Variable

np.random.seed(0)

x = np.linspace(-10., 10., 21)
y = 3 * x

fig, ax = plt.subplots(figsize=(10, 10))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(AutoMinorLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(10))
ax.grid()
ax.grid(which='minor', color='#CCCCCC', linestyle=':')
# plt.gca().set_aspect("equal")


ax.scatter(x, y)

to_var = np.vectorize(lambda val: Variable(val))
to_val = np.vectorize(lambda var: Variable(var))

xs = to_var(x)
xs * Variable(3)
print(xs)
plt.show()
