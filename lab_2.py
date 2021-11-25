
import random
import matplotlib.pyplot as plt
import math
from enum import Enum
from scipy.optimize import minimize, least_squares, curve_fit


def func_x3(arg):

    return math.pow(arg, 3)


def func_abs(arg):

    return math.fabs(arg - 0.2)


def func_sin(arg):

    r_val = arg * math.sin(1/arg)

    return r_val


def one_dimens_exhaustive(l_bound: float, r_bound: float, eps: float, function):
    n: int = int((r_bound - l_bound) / eps)
    delta: float = (r_bound - l_bound) / n  # equals to eps in this case

    global iterations, f_counts

    assert iterations == 0 and f_counts == 0, "iterations and f_counts must equal to zero"

    f_cur: float
    f_min: float = function(l_bound)
    x_min: float = l_bound

    x_arg: float = delta  # first step is made here

    iterations = f_counts = 1

    while x_arg <= r_bound:
        iterations += 1
        f_cur = function(x_arg)
        f_counts += 1
        if f_cur < f_min:
            f_min = f_cur
            x_min = x_arg
        x_arg += delta

    return x_min


def one_dimens_dichotomy_recurse(l_bound: float, r_bound: float, eps: float, function):

    if math.fabs(r_bound - l_bound) < eps:
        return (r_bound + l_bound) / 2  # return average

    delta: float = eps / 2

    global iterations, f_counts

    x1: float = (r_bound + l_bound - delta) / 2
    x2: float = (r_bound + l_bound + delta) / 2

    if function(x1) <= function(x2):
        a1 = l_bound
        b1 = x2
    else:
        a1 = x1
        b1 = r_bound

    iterations += 1
    f_counts += 2

    return one_dimens_dichotomy_recurse(a1, b1, eps, function)


def one_dimens_dichotomy(l_bound: float, r_bound: float, eps: float, function):

    global iterations, f_counts

    assert iterations == 0 and f_counts == 0, "iterations and f_counts must equal to zero"

    return one_dimens_dichotomy_recurse(l_bound, r_bound, eps, function)


class MoveType(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


def one_dimens_golden_sect_recurse(l_bound: float, r_bound: float, eps: float, prev_move: MoveType, f_saved: float, x_saved: float,  function):

    if math.fabs(r_bound - l_bound) < eps:
        return (r_bound + l_bound) / 2  # return average

    delta: float = (3 - math.sqrt(5)) / 2 * (r_bound - l_bound)
    global iterations, f_counts

    x1: float
    x2: float
    f1: float
    f2: float

    saved_x: float
    saved_f: float
    move_type: MoveType

    if prev_move == MoveType.NONE:  # first iteration
        x1 = l_bound + delta
        x2 = r_bound - delta
        f1 = function(x1)
        f2 = function(x2)
        f_counts += 2

    elif prev_move == MoveType.RIGHT:
        x2 = x_saved
        f2 = f_saved
        x1 = l_bound + delta
        f1 = function(x1)
        f_counts += 1


    elif prev_move == MoveType.LEFT:
        x1 = x_saved
        f1 = f_saved
        x2 = r_bound - delta
        f2 = function(x2)
        f_counts += 1


    if f1 <= f2:
        a1 = l_bound
        b1 = x2
        # prepare info for next iteration
        move_type = MoveType.RIGHT
        saved_x = x1
        saved_f = f1

    else:
        a1 = x1
        b1 = r_bound
        # prepare info for next iteration
        move_type = MoveType.LEFT
        saved_x = x2
        saved_f = f2

    iterations += 1

    return one_dimens_golden_sect_recurse(a1, b1, eps, move_type, saved_f, saved_x,  function)


def one_dimens_golden_sect(l_bound: float, r_bound: float, eps: float, function):

    global iterations, f_counts

    assert iterations == 0 and f_counts == 0, "iterations and f_counts must equal to zero"

    return one_dimens_golden_sect_recurse(l_bound, r_bound, eps, MoveType.NONE, 0., 0.,  function)


def gen_rand_vec(vec_len):
    global x_list, y_list

    alpha = random.uniform(0., 1.)
    beta = random.uniform(0., 1.)

    y_list.clear()
    x_list.clear()

    for it in range(vec_len):
        x: float = float(it / vec_len)
        # print(f"x: {x}")
        delta = random.gauss(0, 1)
        # print(f"delta: {delta}")
        y = alpha * x + beta + delta
        y_list.append(y)
        x_list.append(x)


def gen_graph(vec_x, a: float, b: float, function):

    y_values = []

    for x in vec_x:
        y_values.append(function(x, a, b))

    return y_values


def linear_ap_func(x: float, a: float, b: float):
    return a * x + b


def rational_ap_func(x: float, a: float, b: float):
    return a / (1. + b * x)


def D_func(a: float, b: float, function):
    global x_list, y_list

    D_sum = 0.

    for it in range(len(x_list)):
        D_sum += (function(x_list[it], a, b) - y_list[it])*(function(x_list[it], a, b) - y_list[it])

    return D_sum


def D_Lin(ar):
    return D_func(ar[0], ar[1], linear_ap_func)


def D_Rat(ar):
    return D_func(ar[0], ar[1], rational_ap_func)


def two_dimens_exhaustive(l_bound_1: float, r_bound_1: float, l_bound_2: float, r_bound_2: float, eps: float, function):
    n: int = int((r_bound_1 - l_bound_1) / eps)
    delta_1: float = (r_bound_1 - l_bound_1) / n  # equals to eps in this case
    delta_2: float = (r_bound_2 - l_bound_2) / n  # equals to eps in this case

    f_cur: float
    x1_min: float = l_bound_1
    x2_min: float = l_bound_2

    global a_min, b_min

    a_min = b_min = 0

    x1_arg: float = l_bound_1
    x2_arg: float = l_bound_2

    f_min: float = D_func(x1_arg, x2_arg, function)

    while x1_arg <= r_bound_1:
        x2_arg = l_bound_2
        while x2_arg <= r_bound_2:
            f_cur = D_func(x1_arg, x2_arg, function)
            # print(f" x1:{x1_arg}  x2:{x2_arg}  f_cur:{f_cur} f_min:{f_min}")
            if f_cur < f_min:
                f_min = f_cur
                x1_min = x1_arg
                x2_min = x2_arg
            x2_arg += delta_2

        x1_arg += delta_1

    a_min = x1_min
    b_min = x2_min

    return f_min


def two_dimens_gauss(l_bound_1: float, r_bound_1: float, l_bound_2: float, r_bound_2: float, eps: float, function):
    n: int = int((r_bound_1 - l_bound_1) / eps)
    delta_1: float = (r_bound_1 - l_bound_1) / n  # equals to eps in this case
    delta_2: float = (r_bound_2 - l_bound_2) / n  # equals to eps in this case

    f_cur: float
    x1_min: float = l_bound_1
    x2_min: float = l_bound_2

    global a_min, b_min

    a_min = b_min = 0

    # Initial suppose
    x1_arg: float = (r_bound_1 - l_bound_1) / 2
    x2_arg: float = (r_bound_2 - l_bound_2) / 2

    f_min: float = D_func(x1_arg, x2_arg, function)

    while True:
        x1_prev = x1_arg  # store the previous value
        x1_arg = l_bound_1
        f_min = D_func(x1_arg, x2_arg, function)
        while x1_arg <= r_bound_1:
            f_cur = D_func(x1_arg, x2_arg, function)
            # print(f" x1:{x1_arg}  x2:{x2_arg}  f_cur:{f_cur} f_min:{f_min}")
            if f_cur < f_min:
                f_min = f_cur
                x1_min = x1_arg
            x1_arg += delta_1

        x1_arg = x1_min
        if math.fabs(x1_arg - x1_prev) < eps:
            break

        x2_prev = x2_arg  # store the previous value
        x2_arg = l_bound_2
        f_min = D_func(x1_arg, x2_arg, function)
        while x2_arg <= r_bound_2:
            f_cur = D_func(x1_arg, x2_arg, function)
            # print(f" x1:{x1_arg}  x2:{x2_arg}  f_cur:{f_cur} f_min:{f_min}")
            if f_cur < f_min:
                f_min = f_cur
                x2_min = x2_arg
            x2_arg += delta_2

        x2_arg = x2_min
        if math.fabs(x2_arg - x2_prev) < eps:
            break

    a_min = x1_arg
    b_min = x2_arg

    return f_min


print("Type 1 to worl with one-dimensional methods (default)")
print("Type 2 to work with multidimensional methods")

choice = input("> ")

epsilon: float = 0.001

if choice != "2":

    iterations: int
    f_counts: int
    x_min: float
    epsilon = 0.001

    def reset_vals():
        global iterations, f_counts
        iterations = f_counts = 0


    reset_vals()
    x_min = one_dimens_exhaustive(0., 1., epsilon, func_x3)
    print(f"Eshaustive method for x3:      x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

    reset_vals()
    x_min = one_dimens_exhaustive(0., 1., epsilon, func_abs)
    print(f"Eshaustive method for abs:     x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

    reset_vals()
    x_min = one_dimens_exhaustive(0.01, 1., epsilon, func_sin)
    print(f"Eshaustive method for sin:     x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")


    reset_vals()
    x_min = one_dimens_dichotomy(0., 1., epsilon, func_x3)
    print(f"Dichotomy method for x3:       x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

    reset_vals()
    x_min = one_dimens_dichotomy(0., 1., epsilon, func_abs)
    print(f"Dichotomy method for abs:      x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

    reset_vals()
    x_min = one_dimens_dichotomy(0.01, 1., epsilon, func_sin)
    print(f"Dichotomy method for sin:      x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")


    reset_vals()
    x_min = one_dimens_golden_sect(0., 1., epsilon, func_x3)
    print(f"Golden method section for x3:  x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

    reset_vals()
    x_min = one_dimens_golden_sect(0., 1., epsilon, func_abs)
    print(f"Golden method section for abs: x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

    reset_vals()
    x_min = one_dimens_golden_sect(0.01, 1., epsilon, func_sin)
    print(f"Golden method section for sin: x_min - {x_min:1.4f} iterations - {iterations:4},  function counts - {f_counts:4}")

else:

    y_list = []
    x_list = []
    data_len: int = 100
    a_min: float
    b_min: float
    epsilon = 0.001

    point_size = 2.0

    random.seed(data_len)
    gen_rand_vec(data_len)

    # Prepare plots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.plot(x_list, y_list, "ro", markersize=point_size)
    ax1.set_xlabel("Linear approximant", fontsize=14)
    ax2.plot(x_list, y_list, "ro", markersize=point_size)
    ax2.set_xlabel("Rational approximant", fontsize=14)

    # Count for Exhaustive method

    f_min = two_dimens_exhaustive(0., 1., 0., 1., epsilon, linear_ap_func)
    print(f"Linear approximant exhaustive   a:{a_min:1.4f} b:{b_min:1.4f} (D:{f_min:1.4f})")
    ax1.plot(x_list, gen_graph(x_list, a_min, b_min, linear_ap_func), color='blue', label='Exhaustive')


    f_min = two_dimens_exhaustive(0., 1., 0., 1., epsilon, rational_ap_func)
    print(f"Rational approximant exhaustive a:{a_min:1.4f} b:{b_min:1.4f} (D:{f_min:1.4f})")
    ax2.plot(x_list, gen_graph(x_list, a_min, b_min, rational_ap_func), color='blue', label='Exhaustive')

    # Count for Gauss method

    f_min = two_dimens_gauss(0., 1., 0., 1., epsilon, linear_ap_func)
    print(f"Linear approximant gauss        a:{a_min:1.4f} b:{b_min:1.4f} (D:{f_min:1.4f})")
    ax1.plot(x_list, gen_graph(x_list, a_min, b_min, linear_ap_func), color='yellow', label='Gauss')


    f_min = two_dimens_gauss(0., 1., 0., 1., epsilon, rational_ap_func)
    print(f"Rational approximant gauss      a:{a_min:1.4f} b:{b_min:1.4f} (D:{f_min:1.4f})")
    ax2.plot(x_list, gen_graph(x_list, a_min, b_min, rational_ap_func), color='yellow', label='Gauss')

    # Count Nelder-Mead

    (a_min, b_min) = minimize(D_Lin, [0, 0], method='Nelder-Mead', tol=0.001, options={'disp': True}).x
    print(f"Linear approximant nelder-mead   a:{a_min:1.4f} b:{b_min:1.4f} (D:{f_min:1.4f})")
    ax1.plot(x_list, gen_graph(x_list, a_min, b_min, linear_ap_func), color='green', label='Nelder-Mead')

    (a_min, b_min) = minimize(D_Rat, [0, 0], method='Nelder-Mead', tol=0.001, options={'disp': True}).x
    print(f"Rational approximant nelder-mead a:{a_min:1.4f} b:{b_min:1.4f} (D:{f_min:1.4f})")
    ax2.plot(x_list, gen_graph(x_list, a_min, b_min, rational_ap_func), color='green', label='Nelder-Mead')

    ax1.legend()
    ax2.legend()
    plt.show()

