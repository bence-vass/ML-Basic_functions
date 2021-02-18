import math
from collections import defaultdict


def binary_fn_protection(func):
    def check(self, other):
        if self.__class__.__name__ == other.__class__.__name__:
            return func(self, other)
        else:
            raise ValueError('Arguments must be from the same class')

    return check


class Node:
    def __init__(self, value, local_gradients=(), cache=None):
        self._local_gradients = local_gradients
        self._value = value
        self._cache = cache

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def local_gradients(self):
        return self._local_gradients

    @local_gradients.setter
    def local_gradients(self, value):
        if value is None:
            self._local_gradients = ()
        else:
            self._local_gradients = value

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    def __repr__(self):
        return 'NodeObj(' + str(self.value) + ')'

    @binary_fn_protection
    def __mul__(self, other):
        val = self.value * other.value
        local_grads = (
            (self, other.value),
            (other, self.value)
        )
        return Node(val, local_grads)

    @binary_fn_protection
    def __truediv__(self, other):
        val = self.value / other.value
        local_grads = (
            (self, 1 / other.value),
            (other, (-1. * self.value) / (other.value * other.value))
        )
        return Node(val, local_grads)

    @binary_fn_protection
    def __add__(self, other):
        val = self.value + other.value
        local_grads = (
            (self, 1),
            (other, 1)
        )
        return Node(val, local_grads)

    @binary_fn_protection
    def __sub__(self, other):
        val = self.value - other.value
        local_grads = (
            (self, 1),
            (other, -1)
        )
        return Node(val, local_grads)

    @staticmethod
    def sin(node):
        val = math.sin(node.value)
        local_grads = (
            (node, math.cos(node.value)),
        )
        return Node(val, local_grads)

    @staticmethod
    def cos(node):
        val = math.cos(node.value)
        local_grads = (
            (node, -1 * math.sin(node.value)),
        )
        return Node(val, local_grads)

    def tape(self, wrt, refresh_cache=False):
        if not refresh_cache and self.cache:
            return self.cache[wrt]
        else:
            gradients = defaultdict(lambda: 0)

            def calculate_graph(node, path_value):
                for n, e in node.local_gradients:
                    new_path_value = e * path_value
                    gradients[n] += new_path_value
                    calculate_graph(n, new_path_value)

            calculate_graph(self, path_value=1)
            self.cache = gradients

            return gradients[wrt]


x = Node(2.)
y = Node(3.)
z = Node.sin(y) * y

dz_dx = z.tape(y)
print(dz_dx)
