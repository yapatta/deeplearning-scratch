if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from dezero.utils import plot_dot_graph
    from dezero import Variable
    import dezero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward(retain_grad=True)
print(x.grad)
