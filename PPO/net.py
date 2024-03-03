from network import MLP
import numpy as np
import tensorflow as tf
    
inp = np.random.random((1,8))
a = MLP(8, 1)
print(a(inp))

