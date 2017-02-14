# Patrick Komiske, 2017, MIT
#
# A collection of functions that can be useful for a variety of tasks
# across different projects.

import sys, os
import numpy as np


def parg(arg):

    """ Useful function for making single elements into a one-element list,
    i.e. for easy iteration. """

    if type(arg) != list:
        return [arg]
    else:
        return arg 


def fprint(s):

    """ A simple print function which takes a string and prints it to stdout
    immediately. """

    sys.stdout.write(s)
    sys.stdout.flush()


def to_categorical(vector, num_cat):

    """ Takes an a vector of class labels and returns a one hot encoding
    where for each sample the label is a list of zeros with a one in the 
    position corresponding to the class label. """

    return np.asarray([[1 if x == n else 0 for n in range(num_cat)] \
                                                    for x in vector])


def get_unique_file_name(path, filename, suffix = ''):

    """ A function for ensuring unique filenames for saving files. Takes a path,
    filename, and suffix and returns path/filename_index.suffix with the 
    guarantee that this file name is unique in the intended directory. Index is 
    incrementing starting from 0. """

    counter = 0
    os.makedirs(path, exist_ok = True)
    files = os.listdir(path)
    while counter < 10000:
        trial_name = filename + '_{}'.format(counter)
        if trial_name + suffix not in files:
            full_name = trial_name + suffix
            return full_name
        else:
            counter += 1
    raise NameError("Could not find unique filename for {} in {}"
                    .format(filename, path))


def save_model(model, name, path = '../models'):

    """ A function for saving a Keras model. Safely ensures that directories 
    exist before saving.

    model: the Keras model to be saved
    name: the name of the model
    path: where to save the file
    """

    os.makedirs(path, exist_ok = True)
    model.save(os.path.join(path, name))


def data_split(*args, val_frac = .1, test_frac = .1):

    """ A function to split an arbitrary number of arrays into train, 
    validation, and test sets. If val_frac = 0 or test_frac=0, then we 
    don't split any events into the validation set. If exactly two arguments are given 
    (an "X" and "Y") then we return (X_train, Y_train, [X_val, Y_val], 
    [X_test, Y_test]), otherwise lists corresponding to train, [val], [test] 
    splits are returned with entry i corresponding to argument i. Note that
    all arguments must have the same number of samples otherwise an exception
    will be thrown.

    val_frac: fraction of all samples to put in the val dataset. zero means we 
              don't return a val dataset at all
    test_frac: fraction of all samples to put in the test dataset.
    """

    # ensure proper input
    assert 0 <= val_frac < 1.0, 'val_frac invalid'
    assert 0 <= test_frac < 1.0, 'test_frac invalid'
    assert 0 < val_frac + test_frac < 1.0, 'val_frac + test_frac invalid'

    # confirm that all arguments have the same number of samples
    if len(args) == 0:
        raise RuntimeError('Need to pass at least one argument to data_split')
    n_samples = len(args[0])
    for arg in args[1:]:
        if len(arg) != n_samples:
            raise AssertionError('Args to data_split have different length')

    perm = np.random.permutation(np.arange(n_samples, dtype = int))
    
    if num_test == 0:
        num_test = num_val
        num_val = 0

    num_test = int(n_samples * test_frac)
    num_val = int(n_samples * val_frac)
    num_train = n_samples - num_val - num_test

    # if we're doing the conventional train-val-test-split
    if len(args) == 2:
        X_train = args[0][perm[:num_train]]
        Y_train = args[1][perm[:num_train]]
        X_test  = args[0][perm[num_train:num_train+num_test]]
        Y_test  = args[1][perm[num_train:num_train+num_test]]
        if num_val > 0:
            X_val = args[0][perm[-num_val:]]
            Y_val = args[1][perm[-num_val:]]
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
        else:
            return X_train, Y_train, X_test, Y_test
    else:
        train = [arg[perm[:num_train]] for arg in args]
        test = [arg[perm[num_train:num_train+num_test]] for arg in args]
        if num_val > 0:
            val = [arg[perm[-num_val:]] for arg in args]
            return train, val, test
        else:
            return train, test
