import numpy as np
import multiprocessing
import time
import pandas as pd
import os


def execution_time(func_):
    def wrapper(*args, **kwargs):
        begin = time.time()
        res_fn = func_(*args, **kwargs)
        end = time.time()
        t_in_sec = end - begin
        print("function:", func_.__name__, ',', 'num_of_points:', str(args[0]), ',', 'time:', t_in_sec, 'seconds')
        return res_fn, t_in_sec

    return wrapper


def generate_one_sample():
    point = np.random.random((2,))
    distance_from_origo = np.sqrt(point[0] ** 2 + point[1] ** 2)
    is_in = distance_from_origo <= 1
    return is_in


def get_results_based_on_realizations(results):
    mean = np.mean(results)
    pi_estimated = mean * 4
    std = np.std(results)
    diff_vs_th = mean * 4 - np.pi
    return {'pi_estimated': pi_estimated,
            'diff_vs_th': diff_vs_th,
            'sample_mean': mean, 'sample_std': std
            }


@execution_time
def estimate_pi_base(num_of_points):
    results = []
    for i in range(num_of_points):
        is_in = generate_one_sample()
        results.append(is_in)
    return get_results_based_on_realizations(results)


@execution_time
def estimate_pi_multiprocess(num_of_points):
    pool = multiprocessing.Pool()
    results = pool.starmap(generate_one_sample, [() for _ in range(num_of_points)])
    return get_results_based_on_realizations(results)


@execution_time
def estimate_pi_array(num_of_points):
    points = np.random.random((2, num_of_points))
    distances = np.sqrt(points[0, :] ** 2 + points[1, :] ** 2)
    results = distances <= 1
    return get_results_based_on_realizations(results)


def calc_std_error_of_sample(num_of_points):
    points = np.random.random((2, num_of_points))
    distances = np.sqrt(points[0, :] ** 2 + points[1, :] ** 2)
    is_in = distances <= 1
    print(np.std(is_in))


# calc_std_error_of_sample(1000000)
# import sys
# sys.exit()


def execute_given_method(method, num_of_points):
    if method == 'base':
        return estimate_pi_base(num_of_points)
    if method == 'multiprocess':
        return estimate_pi_multiprocess(num_of_points)
    if method == 'array':
        return estimate_pi_array(num_of_points)


if __name__ == "__main__":

    df_execution_time = pd.DataFrame(columns=['num_of_points', 'plain', 'multiprocess', 'vectorized'])
    l_nop = [100, 1000, 10000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 25000000, 50000000]
    folder = 'output'
    filename = 'execution_time.csv'
    for num_of_points_used in l_nop:
        result_base = estimate_pi_base(num_of_points_used)
        result_multiprocess = estimate_pi_multiprocess(num_of_points_used)
        result_array = estimate_pi_array(num_of_points_used)
        df_temp = pd.DataFrame([[num_of_points_used, result_base[1], result_multiprocess[1], result_array[1]]],
                               columns=df_execution_time.columns)
        df_execution_time = pd.concat([df_execution_time, df_temp], axis=0)

    df_execution_time.to_csv(os.path.join(folder, filename), index=False)
    print(df_execution_time)
