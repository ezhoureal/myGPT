import math


class Value:
    def __init__(self, data, _children=(), op='', label=''):
        self.data = data
        self._grad = 0
        self._prev = set(_children)
        self._op = op
        self._label = label
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data = {self.data})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self._grad += out._grad
            other._grad += out._grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self._grad += out._grad * other.data
            other._grad += out._grad * self.data
        out._backward = _backward
        return out
    
    def tanh(self):
        t = (2 / (1 + (math.e ** (-2 * self.data)))) - 1
        out = Value(t, (self,), 'tanh')
        def _backward():
            self._grad += out._grad * (1 - t ** 2)
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), 'pow')
        def _backward():
            self._grad += out._grad * other.data * self.data ** (other.data - 1)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self._grad += 0 if out.data == 0 else out._grad
        out._backward = _backward
        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def backward(self):
        visited = set()
        nodes = []
        def build_graph(v: Value):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build_graph(child)
            nodes.append(v)

        build_graph(self)
        self._grad = 1
        for node in reversed(nodes):
            node._backward()

    def zero_grad(self):
        visited = set()

        def traverse(v: Value):
            if v in visited:
                return
            v._grad = 0
            visited.add(v)
            for child in v._prev:
                traverse(child)

        traverse(self)