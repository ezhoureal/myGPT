from manual_grad.grad import Value
class Neuron:
    def __init__(self, dim_in):
        self.w = [Value(0) for _ in range(dim_in)]
        self.b = Value(0)
    
    def __call__(self, x):
        res = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        res = res.tanh()
        return res

class Layer:
    def __init__(self, dim_in, dim_out):
        self.neurons = [Neuron(dim_in) for _ in range(dim_out)]
    
    def __call__(self, x):
        res = [neuron(x) for neuron in self.neurons]
        return res

class MLP:
    def __init__(self, dim_in, dim_outs: list[int]):
        list = [dim_in] + dim_outs
        self.layers = [Layer(list[i], list[i + 1]) for i in range(len(dim_outs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x