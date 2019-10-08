import random
import time
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
EPS = np.finfo(float).resolution

class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi' : np.full(k, 1/k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-6, verbose=True, max_iters=500):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll
        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True

class CMM(MixtureModel):
    def __init__(self, k, ds, bias = []):
        """
        d is a list containing the number of categories for each feature
        bias is a list of tuples with enumerate(bias) = ith feature, (tuples of feature values with p = 0)
        """
        super(CMM, self).__init__(k)
        self.ds = ds[0]
        self.bias = bias
        self.params['alpha'] = [np.random.dirichlet([1]*(self.ds[i] - len(bias[i])), size=k) for i in range(len(self.ds))] if len(bias) >1 else [np.random.dirichlet([1]*d, size = k) for d in self.ds]

        for i, d_vals in enumerate(bias):
            d_vals = sorted(d_vals)
            for d in d_vals:
                self.params['alpha'][i] = np.insert(self.params['alpha'][i], d, 0, axis = 1)

    def e_step(self, data):
        n, d = data.shape
        ell = np.repeat(np.log(self.pi)[None, :], n, axis=0) # exponentiated log likelihood
        for i, alpha in enumerate(self.alpha):
            dummy = pd.get_dummies(data.iloc[:, i]).T.reindex(np.arange(self.ds[i]), fill_value = 0).T
            ell +=  dummy @ np.log(alpha + EPS).T
        p_z = np.exp(ell) 
        self.params['p_x'] = p_z.sum(1)[:, None]
        p_z /= self.params['p_x']
        return (ell * p_z).sum(), p_z

    def m_step(self, data, p_z):
        new_alpha = [None] * len(self.params['alpha'])
        for i, alpha in enumerate(self.params['alpha']):

            dummy = pd.get_dummies(data.iloc[:, i]).T.reindex(np.arange(self.ds[i]), fill_value = 0).T
            exp_counts = p_z.T @ dummy
            new_alpha[i] = exp_counts / (exp_counts.sum(1)[:, None] + EPS)

        return {
            'pi': p_z.mean(0),
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        n_params = self.k - 1
        n_params += sum(p.shape[0] * (p.shape[1] - 1) for p in self.alpha) # count of free parameters
        return self.max_ll - np.log(self.n_train) * n_params / 2

