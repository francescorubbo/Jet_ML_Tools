# Patrick Komiske, MIT, 2017
#
# Function for importing jet images from multiple .npz files into one numpy
# array. Tries to be efficient by allocating all necessary memory at the 
# beginning. Can also read in events assuming a csv style file format with
# an extra newline per event, particles listed one per line, comments 
# beginning with a '#', and lines denoting overal jet properties beginning
# with 'Event #, [jet_rap], [jet_phi], [jet_pt]'.


import numpy as np
import csv
import os
from utils import *

def data_import(data_type, seed_range, path = '', 
                nevents = 10000, img_size = 33, channels = [0]):

    """ Imports data produced by the Events.cc script into python. Note that both
    gluon and quark files must be present for the desired seed range. The gluons 
    are always listed before the quarks. We assume a constant number of events
    per file.

    data_type: Either 'jetimage' or 'event'. These
               respectively return a numpy array of jet images and a list of events with their particle constituents and a separate
               list of the overall jet four-vectors.
    seed_range: a list or other iterable object containing the seeds for each file
    path: path to directory with the files. Defaults to '../events' and 
          '../images' for the type possible data_types.
    nevents: number of events per file. Note that this needs to be constant across
             the files.
    img_size: the image size of the jet images.
    channels: the channels of the jet image to return.
    """

    assert data_type in ['jetimage', 'event'], 'data_type not recognized'

    if len(path) == 0:
        path == '../events' if data_type == 'event' else '../images'

    if data_type == 'jetimage':
        jetimages = np.zeros((2 * nevents * len(seed_range), len(channels), img_size, img_size))
        start = 0
    elif data_type == 'event':
        jets = []
        jet_tots = []

    for particle_type in ['gluon', 'quark']:
        for index in seed_range:
            filename = particle_type + '-' + data_type + '-seed' + str(index)
            if data_type == 'jetimage':
                jetimages[start:start+nevents, channels] = \
                        np.load(os.path.join(path, filename + '.npz'))
                                                ['arr_0'][:, channels]
                start += nevents

            elif data_type == 'event':
                with open(os.path.join(path, filename + '.res'), 'r') as fh:
                    reader = csv.reader(fh)
                    jet = []
                    for row in reader:
                        if len(row) > 0 and '#' in row[0]:
                            continue
                        if len(row) > 0 and 'Event' in row[0]:
                            jet_tots.append(list(map(float, row[1:])))
                        elif len(row) == 0:
                            jets.append(np.asarray(jet))
                            jet = []
                        else:
                            jet.append(list(map(float, row)))

    if data_type == 'jetimage':
        return jetimages
    elif data_type == 'event':
        return jets, jet_tots
