import pickle
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import MMM as p3
import data_description as ddes

output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")

def fit_k(user, Model, data, k_min, k_max, snap_path, verbose, **model_opts):
    """
    Fit the data from k_min to k_max, and save the model to snap_path with the model
    options defined in model_opts
    """
    indices = []
    for k in range(k_min, k_max + 1):
        while True:
            print('Fitting k = %d' % k, end=', ', flush=True)
            model = Model(k, **model_opts)
            if model.fit(data, verbose=False):
                break
            print('bad init; trying again...')
        model_type = Model.__name__.lower()
        msnap_path = os.path.join(snap_path, '%s_k%s.pkl' % (model_type, k))
        with open(msnap_path, 'wb') as f_snap:
            pickle.dump(model, f_snap)
        threshold(user, model, data, k)

def threshold(user, model, data, k):
    """
    Return events with the smallest 5 probabilities and output to model\least_px_*.csv
    """
    threshold = 5 # first 5 smallest probabilities
    p_x = {i: model.params['p_x'][i][0] for i in range(len(model.params['p_x']))}
    probs = sorted(set(p_x.values()))[:threshold]
    indices = [i for i, j in p_x.items() if j in probs]
    alpha = model.params['alpha']
    # Uncomment here to print feature probability
    # for k in range(len(alpha)):
    #     for i in range(len(alpha[k])):
    #         print(alpha[k][i])
    #         print()
    row = []

    for i in indices:
        row.append({'k': k, 'eventId' : data.iloc[i].name, 'raw csv index' : i+2, 'p(x)' : p_x[i]})

    ddes.write_to(os.path.join(output_dir, "least_px_"+user+".csv"), pd.DataFrame(row), append=True, index_col = 'eventId')

    print("Total number of anomalies with the smallest ", threshold, " p(x) = ", len(indices))
    return indices

def plot_ll_bic(ks, lls, bics):
    """
    Plot for bic
    """
    plt.figure()
    plt.plot(ks, lls, label='ll')
    plt.plot(ks, bics, label='BIC')
    plt.title('EM: Maximized LL and BIC vs K')
    plt.xlabel('K')
    plt.ylabel('(penalized) LL')
    plt.legend()
    plt.show()


def get_k(p):
    return int(re.search('_k(\d+)\.pkl', p).group(1))
