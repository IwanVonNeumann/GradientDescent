import numpy as np

from utils import trace_grad_descent, animate_grad_descent_2d


def z(x):
    return 3 * x[0] ** 2 + x[1] ** 2


def dz(x):
    dx0 = 6 * x[0]
    dx1 = 2 * x[1]
    return np.array([dx0, dx1])


x = np.linspace(start=-10, stop=10, num=100, endpoint=True)
y = np.linspace(start=-10, stop=10, num=100, endpoint=True)

X, Y = np.meshgrid(x, y)
Z = z([X, Y])

alpha = 0.005

gd_X, gd_Z = trace_grad_descent(z, dz, x0=np.array([-9., -9.]), alpha=alpha, n_iter=100)

gd_X = np.array(gd_X)
gd_Z = np.array(gd_Z).reshape(-1, 1)

animate_grad_descent_2d(X, Y, Z, min_point=(0, 0), gd_X=gd_X, title='$3x^2 + y^2$, alpha={}'.format(alpha))
