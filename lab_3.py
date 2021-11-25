import random
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares, curve_fit


def F_lin(x, a, b):
    return a * x + b


def F_rac(x, a, b):
    return a / (1. + x * b)


def D1(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0, n):
        s += (F_lin(X[i], ar[0], ar[1]) - Y[i]) ** 2
    return s


def dD1da(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0, n):
        s += (X[i] ** 2) * ar[0] + X[i] * ar[1] - X[i] * Y[i]
    return 2 * s


def dD1db(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0, n):
        s += X[i] * ar[0] + ar[1] - Y[i]
    return 2 * s


def D2(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0, n):
        s += (F_rac(X[i], ar[0], ar[1]) - Y[i]) ** 2
    return s


def dD2da(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0, n):
        s += ar[0] / (ar[1] * X[i] + 1) ** 2 - Y[i] / (1 + X[i] * ar[1])
    return 2 * s


def dD2db(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0, n):
        s += -1 * ar[0] ** 2 * ar[1] / (ar[1] * X[i] + 1) ** 3 - Y[i] * ar[1] * ar[0] / (1 + X[i] * ar[1]) ** 2
    return s


def gradient_descent(df1, df2):
    global precision
    lamA = lamB = 0.001
    x = y = 0
    lst = [1, 2]  # create list of two elements
    iter = 0
    i = precision + 0.1
    while i > precision:
        lst[0] = x
        lst[1] = y
        iter = iter + 1
        x = x - lamA * df1(lst)
        y = y - lamB * df2(lst)
        if abs(lst[0] - x) > abs(lst[1] - y):
            i = abs(lst[0] - x)
        else:
            i = abs(lst[1] - y)
    print('Iterations number =', iter)
    return x, y


def jaco(x):
    vec = [1, 2]
    vec[0] = dD1da(x)
    vec[1] = dD1db(x)
    return vec


def jaco2(x):
    vec = [1, 2]
    vec[0] = dD2da(x)
    vec[1] = dD2db(x)
    return vec


def gen_rand_vec(vec_len):
    global X, Y

    alpha = random.uniform(0., 1.)
    beta = random.uniform(0., 1.)

    Y.clear()
    X.clear()

    for it in range(vec_len):
        x: float = float(it / vec_len)
        delta = random.gauss(0, 1)
        y = alpha * x + beta + delta
        Y.append(y)
        X.append(x)


def gen_graph(vec_x, a: float, b: float, function):

    y_values = []

    for x in vec_x:
        y_values.append(function(x, a, b))

    return y_values


if __name__ == '__main__':
    random.seed()
    sys.setrecursionlimit(20000)

    X = []
    Y = []
    n: int = 100
    precision = 0.001
    a_min: float
    b_min: float

    # part 2
    alpha = random.random()
    beta = random.random()
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    print('alpha=', alpha, 'beta=', beta)

    gen_rand_vec(n)

    ax1.plot(X, Y, "bo", markersize=3.0)
    ax2.plot(X, Y, "bo", markersize=3.0)

    print('Linear approximant')
    ax1.set_xlabel("Linear approximant", fontsize=14)
    print('Gradient descent')
    res1_1 = gradient_descent(dD1da, dD1db)
    ax1.plot([0, 1], [F_lin(0, res1_1[0], res1_1[1]),
                      F_lin(1, res1_1[0], res1_1[1])], color='yellow', label='Gradient Descent')
    print(f"\t\t Current function value: {D1([res1_1[0], res1_1[1]])}")
    print('Conjugate gradient')
    res1_2 = minimize(D1, [0, 0], method='CG', tol=0.001,
                      options={'disp': True}).x
    ax1.plot([0, 1], [F_lin(0, res1_2[0], res1_2[1]),
                      F_lin(1, res1_2[0], res1_2[1])], color='blue', label='Conjugate Gradient')
    print('Newton')
    res1_3 = minimize(D1, [0, 0], method='Newton-CG', jac=jaco, tol=0.001,
                      options={'disp': True}).x
    ax1.plot([0, 1], [F_lin(0, res1_3[0], res1_3[1]),
                      F_lin(1, res1_3[0], res1_3[1])], color='red', label='Newton\'s')
    print('Levenberg')
    res1_4 = curve_fit(F_lin, X, Y, [0, 0], method='lm', gtol=0.001, full_output=True)
    ax1.plot([0, 1], [F_lin(0, res1_4[0][0], res1_4[0][1]),
                      F_lin(1, res1_4[0][0], res1_4[0][1])], color='green', label='Levenberg-Marquardt ')
    print(f"\t Current function value: {D1([res1_4[0][0], res1_4[0][1]])}")

    print(' ')
    print('Rational approximant')
    ax2.set_xlabel("Rational approximant", fontsize=14)
    print('Gradient descent')

    res2_1 = gradient_descent(dD2da, dD2db)
    ax2.plot(X, gen_graph(X, res2_1[0], res2_1[1], F_rac), color='yellow', label='Gradient Descent')
    print(f"\t\t Current function value: {D2([res2_1[0], res2_1[1]])}")
    print('Conjugate gradient')
    res2_2 = minimize(D2, [0, 0], method='CG', tol=0.001,
                      options={'disp': True}).x
    ax2.plot(X, gen_graph(X, res2_2[0], res2_2[1], F_rac), color='blue', label='Conjugate Gradient')
    print('Newton')
    res2_3 = minimize(D2, [0, 0], method='Newton-CG', jac=jaco2, tol=0.001,
                      options={'disp': True}).x
    ax2.plot(X, gen_graph(X, res2_3[0], res2_3[1], F_rac), color='red', label='Newton\'s')
    print('Levenberg')
    res2_4 = curve_fit(F_rac, X, Y, [0, 0], method='lm', gtol=0.001, full_output=True)
    ax2.plot(X, gen_graph(X, res2_4[0][0], res2_4[0][1], F_rac), color='green', label='Levenberg-Marquardt ')
    print(f"\t\t Current function value: {D2([res2_4[0][0], res2_4[0][1]])}")

    ax1.legend()
    ax2.legend()
    plt.show()

