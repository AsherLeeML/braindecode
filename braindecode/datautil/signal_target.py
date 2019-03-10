import random
import numpy as np

class SignalAndTarget(object):
    """
    Simple data container class.

    Parameters
    ----------
    X: 3darray or list of 2darrays
        The input signal per trial.
    y: 1darray or list
        Labels for each trial.
    """
    def __init__(self, X, y, fs=None):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.fs = fs
    
    def get_subset(self, train_sample_nb, mode='pre', ret3=False):
        if mode == 'pre':
            X = self.X[:train_sample_nb]
            y = self.y[:train_sample_nb]
            fs = self.fs

        if mode == 'post':
            X = self.X[-train_sample_nb:]
            y = self.y[-train_sample_nb:]
            fs = self.fs
        if mode == 'rand':
            idx_list = list(range(self.X.shape[0]))
            random.shuffle(idx_list)
            idices = idx_list[:train_sample_nb]
            X = self.X[idices]
            y = self.y[idices]
            fs = self.fs
            if ret3:
                train_X = X
                train_y = y
                div1 = int(len(idx_list)*0.4)
                div2 = int(len(idx_list)*0.2)
                valid_X = self.X[-div1:-div2]
                valid_y = self.y[-div1:-div2]
                test_X = self.X[-div2:]
                test_y = self.y[-div2:]
                return (SignalAndTarget(train_X, train_y, fs),
                        SignalAndTarget(valid_X, valid_y, fs),
                        SignalAndTarget(test_X, test_y, fs))
            else:
                return SignalAndTarget(X, y, fs)


def apply_to_X_y(fn, *sets):
    """
    Apply a function to all `X` and `y` attributes of all given sets.
    
    Applies function to list of X arrays and to list of y arrays separately.
    
    Parameters
    ----------
    fn: function
        Function to apply
    sets: :class:`.SignalAndTarget` objects

    Returns
    -------
    result_set: :class:`.SignalAndTarget`
        Dataset with X and y as the result of the
        application of the function.
    """
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    return SignalAndTarget(X,y)

def get_balanced_sets_splitter(dataset, valid_size=None, mode='post'):
    """

    the input dataset should be a S&T object.
    
    """
    X, y, fs = dataset.X, dataset.y, dataset.fs
    n_samples = y.shape[0]
    nl = np.unique(y)
    n_classes = nl.shape[0]
    if type(valid_size) == float:
        valid_size_per_class = int(valid_size*n_samples)//n_classes
    else:
        valid_size_per_class = valid_size // n_classes
    idx = [np.where(y==l)[0] for l in nl]
    if mode=='post':
        T_idx = [i[:-valid_size_per_class] for i in idx]
        V_idx = [i[-valid_size_per_class:] for i in idx]
    elif mode=='pre':
        T_idx = [i[valid_size_per_class: ] for i in idx]
        V_idx = [i[ :valid_size_per_class] for i in idx]
    elif mode=='rand':
        for i in idx:
            np.random.shuffle(i)
        T_idx = [i[valid_size_per_class: ] for i in idx]
        V_idx = [i[ :valid_size_per_class] for i in idx]

    T_idx = np.concatenate(T_idx)
    V_idx = np.concatenate(V_idx)
    train_X, valid_X = X[T_idx], X[V_idx]
    train_y, valid_y = y[T_idx], y[V_idx]
    train_set = SignalAndTarget(train_X, y=train_y, fs=fs)
    valid_set = SignalAndTarget(valid_X, y=valid_y, fs=fs)
    return train_set, valid_set