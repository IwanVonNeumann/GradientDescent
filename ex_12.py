import numpy as np

from utils import trace_grad_descent, animate_grad_descent


def x_sinx(x):
    return x * np.sin(x)


def dx_sinx(x):
    return np.sin(x) + x * np.cos(x)


X = np.linspace(start=-2, stop=8, num=100, endpoint=True)
Y = x_sinx(X)
dY = dx_sinx(X)

x0 = 2.5
alpha = 0.05

gd_X, gd_Y = trace_grad_descent(f=x_sinx, df=dx_sinx, x0=x0, alpha=alpha, n_iter=100)
animate_grad_descent(X, Y, gd_X=gd_X, gd_Y=gd_Y, title='alpha = {}'.format(alpha))
