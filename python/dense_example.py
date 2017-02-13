from jet_ML_tools import *
from data_import import *

# specify the number of data files to use
n_files = 2

# import existing data
ecfgs, ecfg_specs = data_import('ecfg', range(1, n_files+1), path = sys.argv[1])
lpolys, lpoly_specs = data_import('lpoly', range(1, n_files+1), path = sys.argv[1])

# hyperparameters
ecfg_hps = { 
    'batch_size': 128,
    'nb_epoch': 1,
    'patience': 3,
    'layer_dims': [50, 100, 50],
    'input_dim': ecfgs.shape[1],
    'model_name': 'ECFG_Example'
}

lpoly_hps = {
    'batch_size': 128,
    'nb_epoch': 1,
    'patience': 3,
    'layer_dims': [50, 100, 50],
    'input_dim': lpolys.shape[1],
    'model_name': 'LPoly_Example'
}


# get labels for the images
Y = make_labels(n_files*10000, n_files*10000)

# split the data into train, validation, and test sets
train, val, test = data_split(ecfgs, lpolys,  Y)

# preprocess the data
for i in [0,1]:
    train[i], val[i], test[i] = zero_center(train[i], val[i], test[i])
    train[i], val[i], test[i] = standardize(train[i], val[i], test[i])


def train_model(hps, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    model = dense_net_construct(hps)

    history = model.fit(X_train, Y_train,
                        batch_size = hps['batch_size'],
                        nb_epoch = hps['nb_epoch'],
                        callbacks = [EarlyStopping(monitor = 'val_loss', 
                                                   patience = hps['patience'], 
                                                   verbose = 1, 
                                                   mode = 'auto')],
                        validation_data = (X_val, Y_val))

    # get a unique name to save the model as
    name = get_unique_file_name('../models', hps['model_name'], '.h5')
    model.save(join('../models', name))

    # construct ROC curve
    qe, ge = ROC_from_model(model, X_test, Y_test)
    save_ROC(model, X_test, Y_test, name)

    # plot other ROC curves
    plot_inv_ROC(qe, ge)
    plot_SI(qe, ge)

train_model( ecfg_hps, train[0], train[2], val[0], val[2], test[0], test[2])
train_model(lpoly_hps, train[1], train[2], val[1], val[2], test[1], test[2])

