from pprint import pprint

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------

def raw_examples():
    a, b = 4, 4
    t = a + b

    x = (3/4) * a + (1/2) * b
    y = (1/4) * a + (1/2) * b
    return dict(a=a,b=b,x=x,y=y, t=t)


def raw_examples_v2():
    a, b = 4, 4
    t = a + b

    w_xa = (3/4)
    w_xb = (1/2)

    w_ya = (1/4)
    w_yb = (1/2)

    x = w_xa * a + w_xb * b
    y = w_ya * a + w_yb * b

    return dict(a=a,b=b,x=x,y=y, t=t)

# ----------------------------------------------------------------------------------------------------------------------

def get_data(t, verbose=0):
    a_r = np.arange(1, t)
    b_r = np.arange(1, t)
    
    filtered_indeces_b_r = np.where(b_r % 2 == 0)
    result_b = b_r[filtered_indeces_b_r]
    if verbose == 1 or verbose == 2:
        print(result_b, type(result_b))
    
    filtered_indeces_a_r = np.where(a_r % 2 == 0)
    result_a = a_r[filtered_indeces_a_r]
    if verbose == 1 or verbose == 2:
        print(result_a, type(result_a))

    result_a = np.array(result_a)
    divide_by_2 = lambda t: t / 2
    result_a_tmp = np.array([divide_by_2(xi) for xi in result_a])
    filtered_indeces_result_a = np.where(result_a_tmp % 2 == 0)
    result_a = result_a[filtered_indeces_result_a]
    
    if verbose == 1 or verbose == 2:
        print(result_a, )
    return result_a, result_b


def get_weights():
    w_y1a = (3/4)
    w_y1b = (1/2)

    w_y2a = (1/4)
    w_y2b = (1/2)

    w_y1 = np.array([w_y1a, w_y1b])
    w_y2 = np.array([w_y2a, w_y2b])
    
    w = np.array([w_y1, w_y2])
    return w


def fit_problem(data, w, t, whole_solutions=False, verbose=0):
    a_els, b_els = data
    all_paris = np.transpose([np.tile(a_els, len(b_els)), np.repeat(b_els, len(a_els))])
    sol = []
    for a, b in all_paris:
        x = np.array([a, b]); y = np.dot(w,x)
        if verbose == 2:
            print(a, b, y, y.sum())
        if y.sum() == t:
            if whole_solutions == True:
                whole_numbers = np.floor(y); result_diff = (y - whole_numbers)
                if verbose == 2:
                    print(y, result_diff)
                result_filter = np.all(result_diff == 0)
                if verbose == 2:
                    print(y, result_diff, result_filter)
                if result_filter == True:
                    sol.append([x, y, t])
                    if verbose == 2:
                        print('append', sol)
                    pass
                else:
                    if verbose == 2:
                        print('no append', y, result_diff, result_filter)
            else:
                if verbose == 2:
                    print('append', [x, y, t])
                sol.append([x, y, t])
            pass
        pass
    return np.array(sol)


def solve_problem(t, verbose=0, whole_solutions=False):
    data = get_data(t, verbose)
    w = get_weights()
    sol = fit_problem(data, w, t, whole_solutions, verbose=verbose)
    return sol


def solve_problem_v2(t_min, t_max, verbose=0, whole_solutions=False):
    
    # Input variables
    data = get_data(t=t_max, verbose=0)
    w = get_weights()
    
    # Variables for running loop and storing solutions
    t_attempts = np.arange(t_min, t_max, 2)
    n = len(t_attempts)
    results = []

    for t in t_attempts:
        
        # Prepare variables for solving problem
        a_els, b_els = data
        indxs_a, indxs_b = np.where(a_els[:] <= t) , np.where(b_els[:] <= t)
        data_ = [a_els[indxs_a], b_els[indxs_b]]
        
        sol = fit_problem(data_, w, t, whole_solutions=False, verbose=0)
        if verbose == 1:
            print(t, sol, len(sol))
            pass
        results.append(sol)
        pass
    # return n, results    
    return np.arange(t_min, t_max, 2), results


# ----------------------------------------------------------------------------------------------------------------------
def create_data_frame(raw_data):
    data = []
    for ii, a_raw_data in enumerate(raw_data):
        for jj, (x, y, t) in enumerate(a_raw_data):
            # print(ii, jj, (x, y, t))
            record = np.concatenate([y,x, [t]])
            # print(ii, jj, record)
            data.append(record)
            pass
        pass
    columns = 'x,y,a,b,t'.split(',')
    df = pd.DataFrame(data=data, columns=columns)
    return df
 