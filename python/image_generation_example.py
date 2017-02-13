# Patrick Komiske, MIT, 2017
#
# An example showing how we can read in data from event/particle information, 
# create the jet images, and save them to file.

from data_import import *
from jet_ML_tools import *

# specify number of files to use
n_files = 2

# import the jets
jets, jet_tots = data_import('event', range(1,1 + n_files))

# form the jet images
jet_images = np.asarray([pixelate(jet, nb_chan = 3) for jet in jets])

# save the jet images to file
write_images_to_file('example_jet_images', jet_images)