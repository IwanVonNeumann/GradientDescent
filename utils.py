import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def trace_grad_descent(f, df, x0, alpha=0.01, n_iter=100):
    x = x0
    points_X = [x]
    points_Y = [f(x)]

    for i in range(n_iter):
        x = x - alpha * df(x)
        y = f(x)
        points_X.append(x)
        points_Y.append(y)

    return points_X, points_Y


def animate_grad_descent(X, Y, gd_X, gd_Y, title=''):
    fig = plt.figure(figsize=(6, 4))

    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.plot(X, Y)

    graph, = plt.plot([], [], 'or-')

    def animate(i):
        graph.set_data(gd_X[:i], gd_Y[:i])
        return graph

    ani = FuncAnimation(fig, animate, frames=len(gd_X), interval=200, repeat=True)

    plt.title(title)
    plt.show()

    return ani


def animate_grad_descent_2d(X, Y, Z, min_point, gd_X, title=''):
    fig = plt.figure(figsize=(6, 4))

    ctr = plt.contour(X, Y, Z)

    plt.grid(True, which='both')

    plt.plot(min_point[0], min_point[1], 'bx', markersize=12)

    graph, = plt.plot([], [], 'or-')

    def animate(i):
        graph.set_data(gd_X[:i, 0], gd_X[:i, 1])
        return graph

    ani = FuncAnimation(fig, animate, frames=len(gd_X), interval=200, repeat=True)

    plt.title(title)
    plt.show()

    return ani
