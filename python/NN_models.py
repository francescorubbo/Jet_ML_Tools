# Patrick Komiske, Eric Metodiev, MIT, 2017
#
# Contains two basic Keras models, a convolutional one with three convolutional
# layers and one dense layer and a dense model with an extensible number of 
# layers.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# ensure Theano dimension ordering
K.set_image_dim_ordering('th')

def conv_net_construct(hps, compiled = True):

    nb_conv = hps['nb_conv']
    nb_pool = hps['nb_pool']
    img_size = hps['img_size']
    nb_filters = hps['nb_filters']
    nb_channels = hps['nb_channels']
    nb_neurons = hps['nb_neurons']
    dropout = hps['dropout']
    act = hps.setdefault('act', 'relu')
    out_dim = hps.setdefault('out_dim', 2)

    model = Sequential()
    model.add(Convolution2D(nb_filters[0], nb_conv[0], nb_conv[0], 
                            input_shape = (nb_channels, img_size, img_size), 
                            init = 'he_uniform', border_mode = 'valid')) 
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size = (nb_pool[0], nb_pool[0])))
    model.add(SpatialDropout2D(dropout[0]))

    model.add(Convolution2D(nb_filters[1], nb_conv[1], nb_conv[1], 
                            init='he_uniform', border_mode = 'valid'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(nb_pool[1], nb_pool[1])))
    model.add(SpatialDropout2D(dropout[1]))

    model.add(Convolution2D(nb_filters[2], nb_conv[2], nb_conv[2], 
                            init='he_uniform', border_mode = 'valid')) 
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(nb_pool[2], nb_pool[2])))
    model.add(SpatialDropout2D(dropout[2]))
    
    model.add(Flatten())

    model.add(Dense(nb_neurons))
    model.add(Activation(act))
    model.add(Dropout(dropout[3]))

    model.add(Dense(out_dim))
    model.add(Activation('softmax'))

    if compiled:
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                      metrics = ['accuracy'])
        model.summary()
        return model

    else:
        return model

def dense_net_construct(hps):
    layer_dims = hps['layer_dims']
    input_dim = hps['input_dim']

    model = Sequential()
    for i,dim in enumerate(layer_dims):
        if i == 0:
            model.add(Dense(dim, input_dim = input_dim, init = 'he_uniform'))
        else:
            model.add(Dense(dim, init = 'he_uniform'))
        model.add(Activation(hps.setdefault('act', 'relu')))
   
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                  metrics = ['accuracy'])
    model.summary()

    return model




