# This file loads datasets from grid search 1 or 2 (change code) and processes it as follows:
# load data into dataFrame, insert uniformly distributed time steps
# interpolate over these time steps
# delete any NaN rows
# calculate a mean over all measurements
# plot the final curve/ curves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')


results = [
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-05-21_15-58-15y3c2e2bh/progress.csv',
        'label': {'search1': 'train batch size = 512'}},
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-07-31_16-11-44z3n33690/progress.csv',
        'label': {'search1': 'train batch size = 512'}},
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-08-01_16-47-48hlvuitmy/progress.csv',
        'label': {'search1': 'train batch size = 512', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-08-02_13-19-05cb7o5wz_/progress.csv',
        'label': {'search1': 'train batch size = 512', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-08-04_15-49-52dcnale6t/progress.csv',
        'label': {'search1': 'train batch size = 512'}},  # 107 highscore
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-08-06_10-21-27bczu_jt1/progress.csv',
        'label': {'search1': 'train batch size = 512'}},
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-07-31_11-25-26z1180k8p/progress.csv',
        'label': {'search1': 'train batch size = 512'}},  # 105
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-07-30_08-05-00a1zxu0ne/progress.csv',
        'label': {'search1': 'train batch size = 512'}},
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-07-29_18-37-33103_te2p/progress.csv',
        'label': {'search1': 'train batch size = 512'}},  # 100+
    {'file': '/Users/tas/LaTex/Forschungsarbeit/Experiment/ray_results/APEX_srv_2019-07-29_18-12-51l8nyjfep/progress.csv',
        'label': {'search1': 'train batch size = 512'}},
]


# rewards_mean: 2
# rewards_max: 0
# iterations: 15
# timesteps: 13
# seconds: 20
# find the paths in ray_results:
# find . -name \progress.csv -print | grep DQN
# print(results[0]['file'], results[0]['label']['search1'])

fig = plt.figure()

# generate human expert data
x = np.linspace(0, 5, 100)
y = 0*x+147
plt.plot(x, y, linestyle='--', label='human reference')

# # outer loop for calculating the averages of several curves
# for x in range(0, 3):
#     if x == 0:
#         parameter = 'train batch size = 64'
#     if x == 1:
#         parameter = 'train batch size = 256'
#     if x == 2:
#         parameter = 'lr = 0.01'

# use counter for renaming columns in dataFrame
i = 0

# create empty df for all data
df_all_data = pd.DataFrame()
parameter = 'train batch size = 512'
# inner loop for processing every TimeSeries
for result in results:
    if 'search1' in result['label'] and result['label']['search1'] == parameter:
        print("Processing dataset ", i)

        # read it
        df = pd.read_csv(filepath_or_buffer=result['file'],
                         delimiter=',',
                         header=1,
                         usecols=[2, 20],
                         names=['mean'+str(i), 'S', ]
                         )

        print(df)

        # get the latest timestamp (last row, column "S" where the seconds of progress.csv are stored
        n = int(df['S'].iloc[-1])

        # since the smallest time steps in the measurements are 2.5 to 3 seconds and the
        # largest time steps are something like 30 seconds, an sampling with 1 second
        # large time steps will be more than enough

        # in the following the oversampling_factor is multiplied the number of measurements
        # to create an over/undersampling
        oversampling_factor = 1
        sample_points = np.linspace(0, n, oversampling_factor*n+1)

        # create a df with only sample points, call it only one letter, because there fucks something up..
        df_sample_points = pd.DataFrame(sample_points, columns=list('S'))

        # append new index column: Therefore ignore indexing (because they will not be unique, also do sorting
        # to make it be sorted :-p , also because the standard behavior will be changed and will be unsorted in
        # future versions of pandas..
        df_sampled = df.append(
            df_sample_points, ignore_index=True, sort=True)
        df_sampled = df_sampled.sort_values(by="S")
        # after sorting, interpolate the NaNs between the real sampling points
        df_sampled = df_sampled.interpolate(method='linear', axis=0)
        # let S be the index (for concatenation AND for plotting)
        df_sampled = df_sampled.set_index("S")

        # add plot for each dataframe
        plt.plot(df_sampled.index.values/60/60,
                 df_sampled['mean'+str(i)], label='__nolegend__')
        # concat the new timeSeries with the others
        df_all_data = pd.concat(
            [df_all_data, df_sampled], axis=1, sort=True, join='outer')
    # increase i for the naming of the columns with the Y-Values
    i += 1
# delete "any" NaNs, that are left and do it inplace (do not make a copy of dataFrame)
df_all_data.dropna(how='any', inplace=True)
# calculate the mean for the 10 series and create a mean column, but skip non-numerical and NaNs
df_all_data['mean'] = df_all_data.mean(
    numeric_only=True, skipna=True, axis=1, )

# add the mean to the plot! Afterwards df_all_data will be deleted and replaced by new data!
plt.plot(df_all_data.index.values/60/60,
         df_all_data['mean'], label='Ape-X mean')
print(df_all_data)

# Now modify Matplotlib for a consistens scaling and view
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, 4, 0, 180))
x1, x2 = plt.xlim()
plt.xlim([0, x2])
y1, y2 = plt.ylim()
plt.ylim([0, y2])

plt.xlabel("time in hours")
plt.ylabel("mean reward")

plt.legend()
plt.show()
