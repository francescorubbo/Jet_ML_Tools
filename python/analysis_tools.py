# Patrick Komiske, 2017, MIT
#
# A collection of functions for analyzing the output of a binary classifier.
# Includes functions for constructing ROC curve, plotting them, saving them,
# and computing derived curves from them.


import numpy as np
import pickle
import os
import matplotlib
#if matplotlib.get_backend() != 'agg':
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def ROC_from_model(model, X_test, Y_test, num_points = 1000):

    """ Computes the ROC curve using a given Keras model and X_test, Y_test
    arrays.

    model: A trained Keras model
    X_test: X values in the test set
    Y_test: Y values in the test set
    num_points: the number of points to construct the curve out of
    """

    # get model output as probability of being a quark
    quark_prob = model.predict_proba(X_test, verbose = 0)#[:,1]
    labels = Y_test[:,1]
    quark_prob = quark_prob.flatten()
    print(quark_prob.shape)
    gluon_eff, quark_eff  = abstract_ROC(quark_prob, labels, num_points)
    return quark_eff, gluon_eff

def plot_distribution(model, X_test, Y_test, save_name, num_points = 1000):
    quark_prob = model.predict_proba(X_test, verbose = 0)[:,1]
    labels = Y_test[:,1]
    gluon_eff, quark_eff = abstract_ROC(quark_prob, labels, num_points)
    area  = ROC_area(gluon_eff, quark_eff)
    plt.title('Distribution : ' + str(area))
    weights = np.ones_like(quark_prob[labels == 1])/float(len(quark_prob[labels == 1]))
    plt.hist(quark_prob[labels == 1],  histtype='step', weights=weights,  normed=True, label="Signal")
    weights = np.ones_like(quark_prob[labels == 0])/float(len(quark_prob[labels == 0]))
    plt.hist(quark_prob[labels == 0],  histtype='step', weights=weights, normed=True, label="Background")
    plt.xlabel("Probability")
    plt.ylabel("Proportion")
    plt.legend(loc='upper left')
    plt.savefig("../plots/weak_dist/" + save_name + "_dist.pdf")
    plt.close()

def abstract_ROC(classifier, labels, num_points = 1000):

    """ A general function for computing a ROC curve for a classifier between
    two classes. Class 0 is assumed to be more likely when the value of the classifier is smaller and class 1 is assumed to be more likely when the 
    value of the classifier is larger.

    classifier: list, values of the classifier for each sample
    labels: list, class labels for each sample
    num_points: number of thresholds to evaluate when constructing the ROC curve
    """ 

    # array for holding predictions based on the threshold
    preds_by_thresh = np.zeros((num_points, len(labels)))

    anti_labels = 1 - labels
    class1_count = np.sum(labels)
    class0_count = len(labels) - class1_count

    min_thresh = np.min(classifier)
    max_thresh = np.max(classifier)

    # iterate through linspace of thresholds
    for i, thresh in enumerate(np.linspace(min_thresh, max_thresh, num_points)):

        # make predictions based on classifier and threshold
        preds_by_thresh[i, classifier > thresh] = 1

    # compute fraction of class 1 we got correct for each threshold
    class1_eff = np.dot(preds_by_thresh, labels)/class1_count

    # compute fraction of class 0 we got correct for each threshold
    class0_eff = 1 - np.dot(preds_by_thresh, anti_labels)/class0_count
    
    return class0_eff, class1_eff


def save_ROC(model, X_test, Y_test, name, num_points = 1000, 
             plot = True, path = '../plots', show = True):

    """ Constructs a ROC curve from a model, plots it if desired, and saves
    the quark and gluon efficiencies to file using pickle. 

    model: A trained Keras model
    X_test: X values in the test set
    Y_test: Y values in the test set
    name: Base file name to be used for both the ROC data and the plot
    num_points: the number of points to construct the curve out of
    plot: if True, saves a plot of the ROC curve
    path: The directory where both the ROC curve and the plot should be saved
    """

    # ensure that the directory we're trying to save to exists
    os.makedirs(path, exist_ok = True)

    quark_eff, gluon_eff = ROC_from_model(model, X_test, Y_test, num_points)

    base_name = name.split('.')[0]
    filename = os.path.join(path, base_name)
    with open(filename + '_ROC_data.pickle', 'wb') as f:
        pickle.dump({'quark_eff': quark_eff, 'gluon_eff': gluon_eff}, f)

    print('\nSaving ROC data for {}'.format(base_name))

    if plot:
        plot_ROC(quark_eff, gluon_eff, show = False)
        plt.savefig(filename + '_ROC_plot.pdf', bbox_inches = 'tight')
        if show:
            plt.show()
        print('\nSaving ROC plot for {}'.format(base_name))


def load_ROC(file, path = '../plots'):

    """ Loads ROC data that was saved with save_ROC. """ 

    with open(os.path.join(path, file), 'rb') as f:
        data = pickle.load(f)
        return data['quark_eff'], data['gluon_eff']


def plot_ROC(quark_eff, gluon_eff, color = 'blue', label = '', show = True):

    """ Plots to default axes the ROC curve given by quark_eff and gluon_eff.

    color: the color to use
    label: what to label the curve
    """
    area = ROC_area(quark_eff, gluon_eff)
    label = label +  " : " + str(area)
    print(label)
    plt.plot(quark_eff, gluon_eff, color = color, label = label)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.xlabel('Quark Signal Efficiency')
    plt.ylabel('Gluon Signal Efficiency')
    plt.title('Area Under ROC Curve: {:.5f}'
              .format(ROC_area(quark_eff, gluon_eff)))
    if show:
        plt.show()
    return area


def plot_inv_ROC(quark_eff, gluon_eff, color = 'blue', label = '',
                 xlim = [.1, 1], ylim = [1, 100], show = True):

    """ Plots to default axes the inverse ROC curve given by quark_eff and 
    gluon_eff.

    color: the color to use
    label: what to label the curve
    xlim: the range of x values to show
    ylim: the range of y values to show
    """

    plt.semilogy(*inv_ROC(quark_eff, gluon_eff), color = color, label = label)
    plt.xticks(np.linspace(0,1,11))
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Quark Signal Efficiency')
    plt.ylabel('1 / Gluon Background Rejection')
    if show:
        plt.show()


def plot_SI(quark_eff, gluon_eff, color = 'blue', label = '',
            xlim = [.1, 1], ylim = [0, 3], show = True):
    
    """ Plots to default axes the significance improvement curve given by
    quark_eff and gluon_eff.

    color: the color to use
    label: what to label the curve
    xlim: the range of x values to show
    ylim: the range of y values to show
    """
    area = ROC_area(quark_eff, gluon_eff)
    label = label +  " : " + str(area)
    plt.plot(*SI(quark_eff, gluon_eff), color = color, label = label)
    plt.xticks(np.linspace(0,1,11))
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Quark Signal Efficiency')
    plt.ylabel('Significance Improvement')
    if show:
        plt.show()


def ROC_area(qe, ge):

    """ Compute area under the ROC curve """

    normal_order = qe.argsort()
    area = np.trapz(ge[normal_order], qe[normal_order])
    area = area if area > 0.5 else 1.0 - area
    return area


def SI(quark_eff, gluon_eff, reg = 10**-6):

    """ Computes the significance improvement from quark efficiency and 
    gluon efificiency. """

    return quark_eff, quark_eff/np.sqrt(1 - gluon_eff + reg)


def inv_ROC(quark_eff, gluon_eff, reg = 10**-6):

    """ Computes the function 1/(false gluon rate) from quark efficiency
    and gluon efficiency. """

    return quark_eff, 1/np.sqrt(1 - gluon_eff + reg)


def gr_at_50_qe(quark_eff, gluon_eff):

    """ Computes the gluon rejection at 50% quark efficiency """

    return 1 - gluon_eff[np.argmin(np.abs(quark_eff - 0.5))]

