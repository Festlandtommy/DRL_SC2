# This file loads the datasets for grid search 1 and processes them in the following way:
# load datasets and insert uniform timesteps
# interpolate over timesteps
# delete all left NaN-Rows (only the uniform timesteps will be left afterwards)
# average over all uniform samples of every parameter set
# plot finally the four averaged curves

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


results_tuple = [

]
results = [
    {'file': '/Users/florian/ray_results/DQN101/DQN_srv_2019-07-01_07-15-44thfl65bi/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN102/DQN_srv_2019-07-01_07-21-38cstnun1y/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN103/DQN_srv_2019-07-01_15-10-40woak50mq/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN104/DQN_srv_2019-07-01_15-11-22km8iip_c/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN105/DQN_srv_2019-07-01_22-13-295skioave/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN106/DQN_srv_2019-07-01_22-14-12c96p8b19/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN107/DQN_srv_2019-07-02_09-52-48_22cz_lj/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN108/DQN_srv_2019-07-02_09-53-529uy879tx/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN109/DQN_srv_2019-07-02_16-16-12x3k84ohv/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN110/DQN_srv_2019-07-02_16-17-37o3benz4d/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN111/DQN_srv_2019-07-02_23-42-48tgw8hxkh/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN112/DQN_srv_2019-07-02_23-43-527jbpxyva/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN113/DQN_srv_2019-07-03_12-10-19nhsu9cso/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN114/DQN_srv_2019-07-03_16-56-13zl3asmyn/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN115/DQN_srv_2019-07-03_20-35-481607xz0k/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN116/DQN_srv_2019-07-04_11-34-356e_dbzbn/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN117/DQN_srv_2019-07-04_11-35-40rlcy6chm/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN118/DQN_srv_2019-07-04_20-26-16n3ccsfk2/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN119/DQN_srv_2019-07-04_20-28-32l2o_700b/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN120/DQN_srv_2019-07-05_11-23-099v6szmyu/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN121/DQN_srv_2019-07-05_11-25-00gn42lc7v/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN122/DQN_srv_2019-07-05_22-06-406w8wt3ob/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN123/DQN_srv_2019-07-05_22-08-48fasrvdho/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN124/DQN_srv_2019-07-06_10-27-39mki1r8r4/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN125/DQN_srv_2019-07-06_10-28-39yfo0vzxb/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN126/DQN_srv_2019-07-06_15-51-56bp3a7qua/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN127/DQN_srv_2019-07-06_15-56-4859p1ktmg/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN128/DQN_srv_2019-07-07_11-55-41fxx0e1xx/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN129/DQN_srv_2019-07-07_11-57-008a4hcje7/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN130/DQN_srv_2019-07-07_18-47-33idsmlk4i/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN131/DQN_srv_2019-07-07_18-51-0400m90kqo/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN132/DQN_srv_2019-07-08_09-42-37p57ueswl/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN133/DQN_srv_2019-07-08_09-45-3924ap6ue_/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN134/DQN_srv_2019-07-08_15-03-480ce90twm/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN135/DQN_srv_2019-07-08_15-05-19e6gbbd1m/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN136/DQN_srv_2019-07-08_20-35-034wup5px4/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN137/DQN_srv_2019-07-08_20-36-502kfb58z4/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN138/DQN_srv_2019-07-09_09-27-12phaa6_wy/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN139/DQN_srv_2019-07-09_09-29-15je3wp6l_/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN140/DQN_srv_2019-07-09_17-05-18hrk3xm9i/progress.csv', 'label': {'search1': 'train batch size = 1024', 'search2': 'hodor'}}
]

# rewards_mean: 2
# rewards_max: 0
# iterations: 15
# timesteps: 13
# seconds: 20
# find the paths in ray_results:
# find . -name \progress.csv -print | grep DQN

# generate human expert data
x = np.linspace(0, 5, 100)
y = 0*x+147

# print(results[0]['file'], results[0]['label']['search1'])
fig = plt.figure()

# create empty df for all data
df_all_data = pd.DataFrame()
# use counter for renaming columns
i = 0
for result in results:
    if i == 0 or i == 4:
        print("Processing dataset ", i)

        df = pd.read_csv(filepath_or_buffer=result['file'],
                         delimiter=',',
                         header=1,
                         usecols=[2, 20],
                         names=['mean'+str(i), 'seconds'+str(i), ]
                         )
        n = len(df.index)
        print("n", n)
        # since the smallest time steps in the measurements are 2.5 to 3 seconds and the
        # largest time steps are something like 30 seconds, an sampling with 1 second
        # large time steps will be more than enough

        # in the following the oversampling_factor is multiplied the number of measurements
        # to create an over/undersampling
        oversampling_factor = 1
        sample_points = np.linspace(0, n, oversampling_factor*n+1)

        # create a df with only sample points,call it column 'A' because 'seconds' does not work... the FUCK...
        df_sample_points = pd.DataFrame(sample_points, columns=list('A'))
        df_sample_points = df_sample_points.rename(columns={"A": "seconds"+str(i)})

        # append new index column
        df_sampled = df.append(df_sample_points, ignore_index=True, sort=True)
        print("LEN", len(df_sampled.index))
        df_sampled = df_sampled.sort_values(by="seconds"+str(i))
        df_sampled = df_sampled.interpolate(method='linear', axis=0)
        print(df_sampled)
        df_sampled = df_sampled.set_index('seconds'+str(i))
        #print(df_sampled)
        # df_copy = df_sampled.copy(deep=True)
        df_all_data = pd.concat([df_all_data, df_sampled], axis=1, sort=True, join='outer')
        #df_all_data.dropna(inplace=True, axis='index', how='all')

        # DAT ISSET....
        df_all_data.dropna(inplace=True)
        plt.plot(df_sampled, label='einzeln')

    i += 1

#df_all_data.dropna(how='all', inplace=True)

df_mean = df_all_data.mean(axis=1)
#df_mean.rename(columns={'0': 'mean'})
#df_all_data = pd.concat([df_all_data, df_mean], sort=True, join='inner')
plt.plot(df_mean, label='mean')
print(df_all_data)

print("ITS da mean!")
print(df_mean)

plt.plot(df_all_data, label='all_data')

# scale axes
x1,x2,y1,y2 = plt.axis()
plt.axis((x1, 30000, 0, 180))
x1, x2 = plt.xlim()
plt.xlim([0, x2])
y1, y2 = plt.ylim()
plt.ylim([0, y2])

plt.legend()
plt.show()
