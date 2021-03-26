if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from dezero.utils import sum_to


x = np.array([[1, 2, 3], [4, 5, 6]])
y = sum_to(x, (1, 3))

print(y)

y = sum_to(x, (2, 1))
print(y)
