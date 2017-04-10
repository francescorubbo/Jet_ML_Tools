# Patrick Komiske, Eric Metodiev, MIT, 2017
#
# A simple example showing how we can import existing jet images, train
# a Keras convolutional model, save the model, and plot and save some of 
# the ROC curves.


from jet_ML_tools import *
from data_import import *
import sys

# specify the number of data files to use
n_files = 2

# import existing jet images
jet_images = data_import('jetimage', range(1, n_files + 1))

# hyperparameters
hps = { 
    'batch_size': 128,
    'img_size': 33,
    'nb_epoch': 15,
    'nb_conv': [8,4,4],
    'nb_filters': [64, 64, 64],
    'nb_neurons': 128,
    'nb_pool': [2, 2, 2],
    'dropout': [.25, .5, .5, .5],
    'patience': 3,
    'nb_channels': 1,
    'model_name': 'Conv_Net_Example'
}

# get labels for the images
Y = make_labels(n_files*10000, n_files*10000)

# split the data into train, validation, and test sets
X_train, Y_train, X_val, Y_val, X_test, Y_test = data_split(jet_images, Y)

# preprocess the data
#X_train, X_val, X_test = zero_center(X_train, X_val, X_test)
#X_train, X_val, X_test = standardize(X_train, X_val, X_test)

model = conv_net_construct(hps)

history = model.fit(X_train, Y_train,
                    batch_size = hps['batch_size'],
                    nb_epoch = hps['nb_epoch'],
                    callbacks = [EarlyStopping(monitor = 'val_loss', 
                                               patience = hps['patience'], 
                                               verbose = 1, 
                                               mode = 'auto')],
                    validation_data = (X_val, Y_val))

# get a unique name to save the model as
name = get_unique_file_name('../models', hps['model_name'])
save_model(model, name + '.h5')

# construct ROC curve
qe, ge = ROC_from_model(model, X_test, Y_test)
save_ROC(model, X_test, Y_test, name)

# plot other ROC curves
plot_inv_ROC(qe, ge)
plot_SI(qe, ge)



