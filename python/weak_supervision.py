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

#
#   Loss Function for Weak Supervision
#       - squared-loss for category-level model predictions
#       - assumes that ytrue and ypred are 1-hot encoded
#
def weak_loss_function(ytrue, ypred):
    return K.square((K.sum(ypred[:,1]) - K.sum(ytrue[:,1])))

#
#   Batch generation for training with Keras
#       - samples are assumed to be an array of N (equally sized) bunches of images
#       - outputs are the targets: proportions for weak supervision, labels for strong supervision
#
def weak_data_generator(samples, outputs, batch_size):
    # yield batches from alternating bunches, then repeat
    while True:
        for i in range(int(samples[0].shape[0]/batch_size)):
            for sample, output in zip(samples, outputs):
                yield sample[i*batch_size:(i+1)*batch_size], output[i*batch_size:(i+1)*batch_size]                

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
def weak_train_CNN(data, labels, hps, bunch_fracs = [0.25, 0.75], val_frac = 0.1, learning_rate = 0.001, weak = True):
    
    X_train, Y_train, X_val, Y_val = data_split(data, labels, val_frac = val_frac, test_frac = 0)
    X_bunches, Y_bunches = make_bunches(X_train, Y_train, bunch_fracs, weak = weak)

    CNN_model = conv_net_construct(hps, compiled = False)
    earlystopper = EarlyStopping(monitor="val_loss", patience= hps['patience'])
    
    # the loss functions are: weak_loss_function for weak supervision and categorical_crossentropy for strong supervision
    if weak:
        CNN_model.compile(loss=weak_loss_function, optimizer=Adam(lr=learning_rate), metrics = ['accuracy']) 
    else:
        CNN_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics = ['accuracy'])
        
    CNN_model.summary()
    if weak:
        print('\nDuring proper training, \'acc\' should tend towards {:.3f}\n'
                                    .format(.5+np.mean([abs(.5-x) for x in bunch_fracs])))
    history = CNN_model.fit_generator(generator = weak_data_generator(X_bunches, Y_bunches, hps['batch_size']), 
                        samples_per_epoch = int(X_bunches[0].shape[0]/hps['batch_size'])*hps['batch_size']*len(X_bunches),
                        nb_epoch = hps['nb_epoch'],
                        validation_data = (X_val, Y_val), 
                        callbacks = [earlystopper])
    return CNN_model
    

# 
#   Running an example training case
#       trained once with weak supervision and once with strong supervision
#       training on a small dataset of 20k events
#
if __name__ == '__main__':

    # specify the file inputs
    n_files, n_events_per_file = 2, 10000

    # read in the data
    data = data_import(data_type='jetimage', seed_range=[1,2], path = '../images/')
    labels = to_categorical(np.concatenate((np.zeros(n_events_per_file*n_files),np.ones(n_events_per_file*n_files))), 2)
    data_train, labels_train, data_test, labels_test = data_split(data, labels, val_frac = 0.0, test_frac = 0.1)

    # CNN hyperparameters based on arXiv:1612.01551
    hps =   {   
                'batch_size': 100, 
                'img_size': 33,
                'nb_epoch': 10, 
                'nb_conv': [8,4,4], 
                'nb_filters': [64, 64, 64],
                'nb_neurons': 128, 
                'nb_pool': [2, 2, 2], 
                'dropout': [.25, .5, .5, .5],
                'nb_channels': 1, 
                'patience': 3, 
                'out_dim' : 2
            }

    # some example bunch fractions
    bunch_fracs = [0.3, 0.4, 0.5, .6, .7]

    # train the model, once weakly and once strongly
    for weak in [True, False]:
        CNN_model = weak_train_CNN(data_train, labels_train, hps, bunch_fracs = bunch_fracs, val_frac = 0.1, weak = weak)
        if weak:
            quark_eff_weak, gluon_eff_weak = ROC_from_model(CNN_model, data_test, labels_test)
        else:
            quark_eff_strong, gluon_eff_strong = ROC_from_model(CNN_model, data_test, labels_test)

    plot_ROC(quark_eff_weak, gluon_eff_weak, show = False, label = 'Weakly supervised')
    plot_ROC(quark_eff_strong, gluon_eff_strong, color = 'red', show = False, label = 'Strongly supervised')
    plt.title('')
    plt.legend(loc = 'lower left')
    os.makedirs('../plots', exist_ok = True)
    plt.savefig('../plots/weak-strong-comparison.pdf', bbox_inches = 'tight')
    plt.show()
        







