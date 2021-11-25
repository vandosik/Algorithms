import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit, dual_annealing


def F_rac(x,a,b,c,d):
    return (a*x+b)/(x**2+c*x+d)


def F_rac_vec(x,a,b,c,d):
    return (a*x+b)/(x**2+c*x+d)


def D_func(ar):
    s = 0
    global X
    global Y
    global n
    for i in range(0,n):
        s += (F_rac(X[i],ar[0],ar[1],ar[2],ar[3])-Y[i])**2
    return s


def func_f(x):
    return 1./(x**2-3*x+2)


def gen_rand_vec(vec_len):
    global X, Y

    Y.clear()
    X.clear()

    y: float
    fval:float

    for it in range(vec_len):
        x: float = float(3 * it / vec_len)
        delta = random.gauss(0, 1)
        fval = func_f(x)

        if fval<-100:
            y = -100 + delta
        elif fval>100:
            y = 100 + delta
        else:
            y = fval + delta

        Y.append(y)
        X.append(x)


def gen_graph( par_ar):

    global X
    y_values = []

    for x in X:
        y_values.append(F_rac(x, par_ar[0], par_ar[1], par_ar[2], par_ar[3]))

    return y_values


if __name__ == '__main__':

    print("Type 1 to worl with stochastic and metaheuristic algorithms (default)")
    print("Type 2 to work with Travelling Salesman Problem")

    choice = input("> ")

    if choice != "2":

        # I part

        random.seed()
        sys.setrecursionlimit(20000)

        X = []
        Y = []
        n: int = 1000
        precision = 0.001
        a_min: float
        b_min: float

        # part 2
        alpha = random.random()
        beta = random.random()
        fig1,  ax2 = plt.subplots(1, 1, figsize=(10, 10))

        gen_rand_vec(n)

        ax2.plot(X, Y, "bo", markersize=3.0)

        ax2.set_xlabel("Rational approximant", fontsize=14)

        print('Differential evolution')
        res2_1 = differential_evolution(D_func, [(-3,3), (-3,3), (-3,3),(-3,3)], maxiter=n, disp=True)
        plt.plot(X, gen_graph(res2_1.x), color='yellow', label='Differential evolution')
        print(res2_1)
        print(f"\t\t Current function value: {D_func(res2_1.x)}")

        print('Annealing')
        res2_2 = dual_annealing(D_func, [(-3,3), (-3,3), (-3,3),(-3,3)], maxiter=n)
        plt.plot(X,gen_graph(res2_2.x), color='grey', label='Annealing')
        print(res2_2)
        print(f"\t\t Current function value: {D_func(res2_2.x)}")

        print('Nelder-Mead')
        res2_3 = minimize(D_func, [0.1,0.1,0.1,0.1], method='Nelder-Mead', tol=0.001, options={'disp': True}).x
        plt.plot(X,gen_graph(res2_3), color='red', label='Nelder-Mead')
        print(res2_3)
        print(f"\t\t Current function value: {D_func(res2_3)}")

        print('Levenberg')
        res2_4 = curve_fit(F_rac, X, Y, [0.1,0.1,0.1,0.1], method='lm', gtol=0.001, full_output=True)
        ax2.plot(X, gen_graph(res2_4[0]), color='green', label='Levenberg-Marquardt ')
        print(res2_4[0])
        print(f"\t\t Current function value: {D_func(res2_4[0])}")

        # res2_4 = least_squares(Residuals, [0, 1, 3, 2], args=(X, Y), method='lm', verbose=2).x
        # print(f"\t\t Current function value: {D_func(res2_4)}")

        ax2.legend()
        plt.show()

    else:
        # II part

        def gen_trans_probability(f_delta: float, temp: float):
            return np.exp(-f_delta / temp)

        def is_trans(prob: float):

            val = random.uniform(0., 1.)

            if val <= prob:
                return True
            else:
                return False

        def gen_neighbour(cur_state: list):

            neig_list = cur_state[:]

            first = random.randint(0, len(cur_state)-1)
            second = random.randint(0, len(cur_state)-1)
            # change two items

            tmp = neig_list[first]
            neig_list[first] = neig_list[second]
            neig_list[second] = tmp

            return neig_list

        def calculate_energy(cur_state: list, d_matrix: list):

            route_len = 0

            for it in range(1, len(cur_state)):
                start_p = cur_state[it-1]
                fin_p = cur_state[it]
                route_len += d_matrix[start_p][fin_p]
            # Add way to first city from last
            route_len += d_matrix[cur_state[len(cur_state)-1]][cur_state[0]]
            return route_len


        def gen_graph_way(way: list, coord: list):
            x_data = []
            y_data = []

            for point in way:
                x_data.append(coord[point][0])
                y_data.append(coord[point][1])
            # Add first nodeagain
            x_data.append(coord[way[0]][0])
            y_data.append(coord[way[0]][1])

            return x_data, y_data

        random.seed()
        max_iterations = 1000
        initial_temp = 10.

        with open('matrix.txt') as f:
            matrix = [list(map(int, row.split())) for row in f.readlines()]

        with open('coordinates.txt') as f:
            coordinates = [list(map(float, row.split())) for row in f.readlines()]

        # Gen first way
        cur_way = []
        for i in range(0, len(coordinates)):
            cur_way.append(i)

        first_score = cur_way[:]
        print("First way")
        print(first_score)

        cur_energy = calculate_energy(cur_way, matrix)
        cur_temp = initial_temp
        delta_temp = initial_temp / max_iterations

        # Need to dynamically update the graph
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()

        # Main Loop
        for it in range(max_iterations): # secure for too much iterations

            print(it, cur_energy)

            neighbour = gen_neighbour(cur_way)
            neigh_energy = calculate_energy(neighbour, matrix)

            if neigh_energy < cur_energy:
                cur_energy = neigh_energy
                cur_way = neighbour
            else:
                p = gen_trans_probability(neigh_energy - cur_energy, cur_temp)

                if is_trans(p):
                    cur_energy = neigh_energy
                    cur_way = neighbour
                    # Redraw plot
                    x_arr, y_arr = gen_graph_way(cur_way, coordinates)
                    ax.clear()
                    ax.plot(x_arr, y_arr, "ro", markersize=5.)
                    ax.plot(x_arr, y_arr, "b", markersize=3.)
                    ax.set_xlabel("Temp: " + str(cur_temp), fontsize=14)
                    fig.canvas.draw()

                    time.sleep(0.0001)

            cur_temp -= delta_temp

            if cur_temp <= 0: # this will coincide with normal finish
                break

        plt.ioff()
        plt.close(fig)
        time.sleep(1.)

        best_score = cur_way[:]
        print("Best way found:")
        print(best_score)


        # Prepare final plots
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        x_arr, y_arr = gen_graph_way(first_score, coordinates)
        ax1.plot(x_arr, y_arr, "ro", markersize=5.)
        ax1.plot(x_arr, y_arr, "b", markersize=3.)
        ax1.set_xlabel("First iteration", fontsize=14)

        x_arr, y_arr = gen_graph_way(best_score, coordinates)
        ax2.plot(x_arr, y_arr, "ro", markersize=5.)
        ax2.plot(x_arr, y_arr, "b", markersize=3.)
        ax2.set_xlabel("Received best solution", fontsize=14)

        real_best_solution = [1, 13, 2, 15, 9, 5, 7, 3, 12, 14, 10, 8, 6, 4, 11]  # nodes are 1 to 15, need 0-14
        for it in range(len(real_best_solution)):
            real_best_solution[it] -= 1

        print("Real best way:")
        print(real_best_solution)

        x_arr, y_arr = gen_graph_way(real_best_solution, coordinates)
        ax3.plot(x_arr, y_arr, "ro", markersize=5.)
        ax3.plot(x_arr, y_arr, "b", markersize=3.)
        ax3.set_xlabel("Real best solution", fontsize=14)

        plt.show()