#================================================================================
#
#   Weak Supervision with (grayscale) jet images
#
#   Eric M. Metodiev and Patrick T. Komiske
#   metodiev@mit.edu, pkomiske@mit.edu
#   Massachusetts Institute of Technology, 2017
#
#================================================================================

from jet_ML_tools import *
from data_import import data_import
import random
import os 
from keras.regularizers import l1
from keras.initializers import VarianceScaling
from keras.optimizers import Adamax, Nadam
#CONSTANTS
curr_batch_size = 6456

#
#   Loss Function for Weak Supervision
#       - squared-loss for category-level model predictions
#       - assumes that ytrue and ypred are 1-hot encoded
#
def weak_loss_function(ytrue, ypred):
    return K.square((K.sum(ypred) - K.sum(ytrue[:,1]))/curr_batch_size)

#
#   Batch generation for training with Keras
#       - samples are assumed to be an array of N (equally sized) bunches of images
#       - outputs are the targets: proportions for weak supervision, labels for strong supervision
#
def weak_data_generator(samples, outputs, batch_size):
    # yield batches from alternating bunches, then repeat
    while True:
        for sample, output in zip(samples, outputs):
            yield sample, output

#                
#   Bunch creation with different fractions
#       - takes in labelled Q/G samples and bunch fractions and outputs proportioned bunches
#       - assumes labels Y are one-hot encoded
#       - returns an array of bunches and targets with equal amounts of data per bunch
#
def make_bunches(X, Y, bunch_fracs, weak = True, b_size_input = float('inf')):
    # get numbers of input quark and gluon events, separate the quark and gluon data
    n_gluons, n_quarks = np.sum(Y[:,0]), np.sum(Y[:,1])
    np.random.shuffle(bunch_fracs)
    X_gluon, X_quark = X[Y[:,0]==1], X[Y[:,1]==1]
    X_out, Y_out = [], []
    
    # determine the maximum amount of quark and gluon events per bunch
    f_avg = np.mean(bunch_fracs)
    if (n_quarks/n_gluons > f_avg/(1 - f_avg)):
        bunch_size = int(int(n_gluons + n_gluons * f_avg/(1 - f_avg))/len(bunch_fracs))
        n_gluons_by_bunch = np.array([int(bunch_size * (1 - frac)) for frac in bunch_fracs])
        n_quarks_by_bunch = bunch_size - n_gluons_by_bunch
    else:
        bunch_size = int(int(n_quarks + n_quarks * (1 - f_avg)/f_avg)/len(bunch_fracs))
        n_quarks_by_bunch = np.array([int(bunch_size * frac)  for frac in bunch_fracs])
        n_gluons_by_bunch = bunch_size - n_quarks_by_bunch

    # use the user-specific bunch size if possible
    bunch_size = min([bunch_size, b_size_input])
 
    # set the curr_batch_size global variable
    global curr_batch_size
    curr_batch_size = bunch_size 

    fprint('Using bunch_size = {}\n'.format(bunch_size))
    
    # create each bunch by including the appropriate numbers of quark and gluon events
    gi, qi = 0, 0
    for ng, nq, frac in zip(n_gluons_by_bunch, n_quarks_by_bunch, bunch_fracs):
        perm = np.random.permutation(np.arange(bunch_size)).astype(int)
        X_out.append(np.concatenate((X_gluon[gi:gi+ng], X_quark[qi:qi+nq]))[perm])

        # the targets are: [1-frac,frac] for weak supervision and [g_label, q_label] for strong supervision
        if weak:
            Y_out.append((1-frac)*to_categorical(np.zeros(bunch_size), 2) + frac*to_categorical(np.ones(bunch_size), 2))
        else:
            Y_out.append(to_categorical(np.concatenate((np.zeros(ng), np.ones(nq)))[perm], 2))
        gi, qi = gi + ng, qi + nq

    return X_out, Y_out

#
#   Train the CNN with weak or strong supervision
#       - takes in data and labels from generated events
#       - need to specify the different bunch fractions to use
#       - outputs a trained model
#
def weak_train_CNN(data, labels, hps, bunch_fracs = [0.25, 0.75], val_frac = 0.3, learning_rate = 0.002, save_fname='model_weak', weak=True):
 
    if weak: 
        print("Training Weak Supervision")
        all_X_bunches, all_Y_bunches = make_bunches(data, labels, bunch_fracs)
        all_X_bunches, all_Y_bunches = np.array(all_X_bunches), np.array(all_Y_bunches)

        num_bunches = len(all_X_bunches)
        val_indices = np.array(random.sample(range(num_bunches), int(val_frac*num_bunches)))
        X_val, Y_val = np.vstack(all_X_bunches[val_indices]), np.vstack(all_Y_bunches[val_indices])

        train_indices = [k for k in range(num_bunches) if k not in val_indices]
        X_bunches = np.vstack(all_X_bunches[train_indices])
        Y_bunches = np.vstack(all_Y_bunches[train_indices])

    CNN_model = weak_conv_net_construct(hps, compiled = False)
    earlystopper = EarlyStopping(monitor="val_loss", patience= hps['patience'])
    checkpointer = ModelCheckpoint(save_fname, monitor="val_loss", save_best_only=True)
    if os.path.exists(save_fname):
        print('Weight File Exists. Replacing ...')
        os.remove(save_fname)

    if weak:
        CNN_model.compile(loss=weak_loss_function, optimizer=Nadam(lr=learning_rate), metrics = ['accuracy']) 
    else:
        CNN_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics = ['accuracy'])

    CNN_model.summary()
    print('\nDuring proper training, \'acc\' should tend towards {:.3f}\n'
                                    .format(.5+np.mean([abs(.5-x) for x in bunch_fracs])))
    print('Batch size being used : ', curr_batch_size)
    if weak: 
        history = CNN_model.fit(X_bunches, Y_bunches, batch_size=curr_batch_size,  shuffle='batch', nb_epoch=hps["nb_epoch"],\
            validation_data=(X_val, Y_val), callbacks=[earlystopper, checkpointer])
    else:
         history = CNN_model.fit(data, labels,  shuffle=True, nb_epoch=hps["nb_epoch"],\
        validation_split=val_frac, callbacks=[earlystopper, checkpointer])
    CNN_model.load_weights(save_fname)
    #print("Train Loss : ", CNN_model.evaluate(X_bunches, Y_bunches, batch_size=curr_batch_size))
    #print("Val Loss : ", CNN_model.evaluate(X_val, Y_val, batch_size=curr_batch_size))
    return CNN_model
    

def weak_conv_net_construct(hps, compiled = True):

    nb_conv = hps['nb_conv']
    nb_pool = hps['nb_pool']
    img_size = hps['img_size']
    nb_filters = hps['nb_filters']
    nb_channels = hps['nb_channels']
    nb_neurons = hps['nb_neurons']
    dropout = hps['dropout']
    act = hps.setdefault('act', 'elu')
    out_dim = hps.setdefault('out_dim', 1)
    init = hps['init']
    if init == "var_scaling":
        init = VarianceScaling(scale=0.5)
    reg = 0.0000
    model = Sequential()
    model.add(Convolution2D(nb_filters[0], nb_conv[0], nb_conv[0],
                            input_shape = (nb_channels, img_size, img_size),
                            init = init, border_mode = 'valid', W_regularizer=l1(reg)))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size = (nb_pool[0], nb_pool[0])))
    model.add(SpatialDropout2D(dropout[0]))

    model.add(Convolution2D(nb_filters[1], nb_conv[1], nb_conv[1],
                            init=init, border_mode = 'valid', W_regularizer=l1(reg)))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(nb_pool[1], nb_pool[1])))
    model.add(SpatialDropout2D(dropout[1]))

    model.add(Convolution2D(nb_filters[2], nb_conv[2], nb_conv[2],
                            init=init, border_mode = 'valid', W_regularizer=l1(reg)))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(nb_pool[2], nb_pool[2])))
    model.add(SpatialDropout2D(dropout[2]))

    model.add(Flatten())

    model.add(Dense(nb_neurons))
    model.add(Activation(act))
    model.add(Dropout(dropout[3]))

    model.add(Dense(out_dim))
    model.add(Activation('sigmoid')) #'softmax'))

    if compiled:
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                      metrics = ['accuracy'])
        model.summary()
        return model
    else:
        return model
