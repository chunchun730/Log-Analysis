import glob
import os
import pickle
import MMM as p3
import utils as utils
import pandas as pd
import argparse

def main(user):
    """
    Run EM on user.csv with k from 1 to 6 by default, plot the bic and ll graph
    and generate least_likely_events.csv under model folder as a result of the clustering
    """
    PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(PROJ_DIR, 'models')
    user_dir = os.path.join(PROJ_DIR, 'users')
    data_dir = os.path.join(PROJ_DIR, 'data')
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    data = pd.read_csv(os.path.join(user_dir, user+".csv"), index_col=0)
    ds = [data.max(axis=0)+1]

    CMM_K_MIN_MAX = (1, 6)
    utils.fit_k(user, p3.CMM, data, *CMM_K_MIN_MAX, MODELS_DIR, verbose=True, ds=ds)

    snaps = glob.glob(os.path.join(MODELS_DIR, 'cmm_*.pkl'))
    snaps.sort(key=utils.get_k)
    ks, bics, lls = [], [], []
    for snap in snaps:
        with open(snap, 'rb') as f_snap:
            model = pickle.load(f_snap)
        ks.append(utils.get_k(snap))
        lls.append(model.max_ll)
        bics.append(model.bic)
    utils.plot_ll_bic(ks, lls, bics)

if __name__ == "__main__":
    """
    Run the script to cluster the events
    arg:
        -u email of the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', action='store', required = True,
                        dest='user',
                        help='User email')
    result = parser.parse_args()

    main(result.user)

