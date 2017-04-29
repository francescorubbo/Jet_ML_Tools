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
from weak_model import weak_train_CNN
from optparse import OptionParser

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--n", action="store_true", dest="normalize", default=True)
    parser.add_option("--ne", type="int", dest="nb_epoch", default=50)
    parser.add_option("--init", type="string", dest="init", default="var_scaling")
    parser.add_option("--num_frac", type="int", dest="num_frac", default=50)
    parser.add_option("--act", type="string", dest="act", default='elu')
    parser.add_option("--num_iter", type="int", dest="num_iter", default=10)
    parser.add_option("--data_frac", type="float", dest="data_frac", default=1.0)
    parser.add_option("--learning_rate", type="float", dest="lr", default=5e-4)
    parser.add_option("--save_name", type="string", dest="save_name", default="3_normed_50var_scaling05_weak_crossent_elu_nodp10-1")
    options, args = parser.parse_args()
    normalize = options.normalize
    nb_epoch  = options.nb_epoch
    init = options.init
    num_frac = options.num_frac
    num_iter = options.num_iter
    save_name = options.save_name
    lr = options.lr
    act = options.act
    data_frac = options.data_frac

    print(options)
    # specify the file inputs
    n_files, n_events_per_file = 2, 10000

    # read in the data
    data = data_import(data_type='jetimage', seed_range=[1,2], path = '../images/')
    labels = to_categorical(np.concatenate((np.zeros(n_events_per_file*n_files),np.ones(n_events_per_file*n_files))), 2)
    data_train, labels_train, data_test, labels_test = data_split(data, labels, val_frac = 0.0, test_frac = 0.1)
	
    num_train = len(data_train)
    print("Original Data Length : ", num_train, " Test length : ", len(data_test) )
    num_use  = int(data_frac*num_train)
    indices = np.random.permutation(len(data_train))[:num_use]
    data_train = data_train[indices]
    labels_train = labels_train[indices]
    print("New Data length :", len(data_train))
    
    if normalize:
        data_train, data_test  = zero_center(data_train, data_test)
        data_train, data_test  = standardize(data_train, data_test)
    
    # CNN hyperparameters based on arXiv:1612.01551
    hps =   {   
                'batch_size': 128, 
                'img_size': 33,
                'nb_epoch': nb_epoch, 
                'nb_conv': [8, 4, 4], 
                'nb_filters': [64, 64, 64],
                'nb_neurons': 128, 
                'nb_pool': [2, 2, 2], 
                'dropout': [0, 0, 0, 0, 0], #.25, .5, .5, .5],
                'nb_channels': 1, 
                'init' : init,
                'patience': 5, 
                'out_dim' : 2,
                'act' : act
            }

    # some example bunch fractions
    bunch_fracs = np.linspace(0.1, 1.0, num_frac) 
    aucs = []
    for i in range(num_iter):
        CNN_model = weak_train_CNN(data_train, labels_train, hps, bunch_fracs = bunch_fracs, val_frac = 0.3, learning_rate=lr, save_fname=save_name, weak=True)
        quark_eff_weak, gluon_eff_weak = ROC_from_model(CNN_model, data_test, labels_test)
        area = ROC_area(quark_eff_weak, gluon_eff_weak)
        aucs.append(area)
        print("iter : ", i, " - Area : ", area)
        #if area > 0.8:
        #    curr_save_name = save_name + str(i)
        #    plot_distribution(CNN_model, data_test, labels_test, curr_save_name)
    print('AUCS : ', aucs)
    np.save("aucs/" + save_name + "_aucs", aucs)
    median = np.median(aucs)
    q75, q25 = np.percentile(aucs, [75, 25])
    iqr = q75 - q25
    print("median : ", median, " iqr : ", iqr)



