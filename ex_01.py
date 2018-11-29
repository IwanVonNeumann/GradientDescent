import numpy as np

from utils import trace_grad_descent, animate_grad_descent


def x_squared(x):
    return (1 - x) ** 2 + 2


def dx_squared(x):
    return 2 * x - 2


X = np.linspace(start=-2, stop=4, num=50, endpoint=True)
Y = x_squared(X)
dY = dx_squared(X)

x0 = -2
alpha = 0.005

gd_X, gd_Y = trace_grad_descent(f=x_squared, df=dx_squared, x0=x0, alpha=alpha, n_iter=100)
ani = animate_grad_descent(X, Y, gd_X=gd_X, gd_Y=gd_Y, title='alpha = {}'.format(alpha))
