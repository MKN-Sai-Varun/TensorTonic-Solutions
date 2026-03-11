import numpy as np

def leaky_relu(x, alpha=0.01):
    result = []
    
    for num in x:
        if num > 0:
            result.append(num)
        else:
            result.append(num * alpha)
    
    return np.array(result)