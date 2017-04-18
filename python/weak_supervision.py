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
from optparser import OptionParser

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--n", action="store_true", dest="normalize", default=False)
    parser.add_option("--ne", type="int", dest="nb_epoch", default=1)
    parser.add_option("--init", type="string", dest="init", default="he_normal")
    parser.add_option("--num_frac", type="int", dest="num_frac", default=16)
    parser.add_option("--num_iter", type="int", dest="num_iter", default=1)
    
    options, args = parser.parse_args()
    normalize = options.normalize
    nb_epoch  = options.nb_epoch
    init = options.init
    num_frac = options.num_frac
    num_iter = options.num_iter

    # specify the file inputs
    n_files, n_events_per_file = 2, 10000

    # read in the data
    data = data_import(data_type='jetimage', seed_range=[1,2], path = '../images/')
    labels = to_categorical(np.concatenate((np.zeros(n_events_per_file*n_files),np.ones(n_events_per_file*n_files))), 2)
    data_train, labels_train, data_test, labels_test = data_split(data, labels, val_frac = 0.0, test_frac = 0.1)

    if normalize:
        data_train, data_test  = zero_center(data_train, data_test)
        data_train, data_test  = standardize(data_train, data_test)
    
    # CNN hyperparameters based on arXiv:1612.01551
    hps =   {   
                'batch_size': 128, 
                'img_size': 33,
                'nb_epoch': nb_epoch, 
                'nb_conv': [8,4,4], 
                'nb_filters': [64, 64, 64],
                'nb_neurons': 128, 
                'nb_pool': [2, 2, 2], 
                'dropout': [.25, .5, .5, .5],
                'nb_channels': 1, 
                'init' : init
                'patience': 15, 
                'out_dim' : 2
            }

    # some example bunch fractions
    bunch_fracs = np.linspace(0.1, 1.0, num_frac) 
    aucs = []
    for i in range(num_iter):
        CNN_model = weak_train_CNN(data_train, labels_train, hps, bunch_fracs = bunch_fracs, val_frac = 0.3)
        quark_eff_weak, gluon_eff_weak = ROC_from_model(CNN_model, data_test, labels_test)

    plot_ROC(quark_eff_weak, gluon_eff_weak, show = False, label = 'Weakly supervised')
    plt.title('')
    plt.legend(loc = 'lower left')
    os.makedirs('../plots', exist_ok = True)
    plt.savefig('../plots/weak-strong-comparison.pdf', bbox_inches = 'tight')
    plt.show()
        




