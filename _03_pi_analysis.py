import pandas as pd
import matplotlib.pyplot as plt
import _02_pi_estimation as pi_est
import numpy as np
import scipy.stats as st


# ------ Accuracy ------
# according to CentralLimitTheorem, sample mean is a stochastic variable
# which has normal distribution around theoretical mean (pi)
# with standard error of std / sqrt(n)
# so using error is scaling with 1/sqrt(n)


def plot_sampled_accuracy(l_points, conf_level, random_seed=None, show=True):
    if random_seed is not None:
        np.random.seed(random_seed)
    df_res = pd.DataFrame(
        columns=['num_of_points', 'pi_estimated', 'pi_th', 'diff_vs_th', 'conf_iv_lower', 'conf_iv_upper'])
    for num_of_points in l_points:
        res = pi_est.estimate_pi_base(num_of_points)
        std_error_of_mean = 4 * res[0]['sample_std'] / np.sqrt(num_of_points)
        conf_iv = st.t.interval(alpha=conf_level, df=num_of_points - 1,
                      loc=res[0]['pi_estimated'], scale=std_error_of_mean)
        df_temp = pd.DataFrame(
            [[num_of_points, res[0]['pi_estimated'], np.pi, res[0]['diff_vs_th'], conf_iv[0], conf_iv[1]]],
            columns = df_res.columns
        )
        df_res = df_res.append(df_temp, ignore_index=True)
    df_res.index = df_res['num_of_points']
    df_res[['pi_estimated', 'pi_th', 'conf_iv_lower', 'conf_iv_upper']].plot()
    plt.title('Estimated ' + str(conf_level) + ' confidence interval for pi estimation')
    plt.xlabel('Number of points')
    if show:
        plt.show()
    return df_res


def plot_accuracy_theoretical(l_points, conf_level, show=True):

    df_res = pd.DataFrame(
        columns=['num_of_points', 'pi_th', 'conf_iv_lower', 'conf_iv_upper'])

    std_th = 4 * np.sqrt(np.pi/4*(1-np.pi/4))

    for num_of_points in l_points:
        std_error_of_mean = std_th / np.sqrt(num_of_points)
        conf_iv = st.t.interval(alpha=conf_level, df=num_of_points,
                                loc=np.pi, scale=std_error_of_mean)
        df_temp = pd.DataFrame(
            [[num_of_points, np.pi, conf_iv[0], conf_iv[1]]],
            columns=df_res.columns
        )
        df_res = df_res.append(df_temp, ignore_index=True)
    df_res.index = df_res['num_of_points']
    df_res[['pi_th', 'conf_iv_lower', 'conf_iv_upper']].plot()
    plt.title('Theoretical ' + str(conf_level) + ' confidence interval for pi estimation')
    plt.xlabel('Number of points')
    if show:
        plt.show()
    return df_res


def execute_accuracy_analysis(l_points, conf_level, random_seed=None):
    df_res = plot_sampled_accuracy(l_points, conf_level, random_seed, show=False)
    df_res_th = plot_accuracy_theoretical(l_points, conf_level, show=False)
    return df_res, df_res_th
# l_points = [5000, 10000, 20000, 50000, 100000, 200000, 1000000, 2000000]
# conf_level = 0.98
# execute_accuracy_analysis(l_points, conf_level, random_seed=114)
# plt.show()


# ---------- speedup ----------
df = pd.read_csv('output/execution_time.csv', index_col=0)
print(df)
df['speedup'] = df['plain'] / df['multiprocess']
df['speedup'].plot()
plt.title('Speedup = plain / multiprocess')

df_time_of_1m = df.copy()
for col in ['plain', 'multiprocess', 'vectorized']:
    df_time_of_1m[col] = df_time_of_1m[col]/df_time_of_1m.index*1000000
df_time_of_1m.loc[df_time_of_1m.index[4:]].plot()
plt.title('Execution time (seconds) of 1M points')

plt.show()
