import random
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
    
    def get_subset(self, sample_nb, mode='pre'):
        if mode == 'pre':
            X = self.X[:sample_nb]
            y = self.y[:sample_nb]
            fs = self.fs
            return SignalAndTarget(X, y, fs)
        if mode == 'post':
            X = self.X[-sample_nb:]
            y = self.y[-sample_nb:]
            fs = self.fs
            return SignalAndTarget(X, y, fs)
        if mode == 'rand':
            idices = range(self.X.shape[0])
            random.shuffle(idices)
            idices = idices[:sample_nb]
            X = self.X[idices]
            y = self.y[idices]
            fs = self.fs
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
