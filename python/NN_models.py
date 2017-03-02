from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D
from os.path import basename

def conv_net_construct(hps):

    nb_conv = hps['nb_conv']
    nb_pool = hps['nb_pool']
    img_size = hps['img_size']
    nb_filters = hps['nb_filters']
    nb_channels = hps['nb_channels']
    nb_neurons = hps['nb_neurons']
    dropout = hps['dropout']

    model = Sequential()
    model.add(Convolution2D(nb_filters[0], nb_conv[0], nb_conv[0], 
                            input_shape = (nb_channels, img_size, img_size), 
                            init = 'he_uniform', border_mode = 'valid')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (nb_pool[0], nb_pool[0])))
    model.add(SpatialDropout2D(dropout[0]))

    model.add(Convolution2D(nb_filters[1], nb_conv[1], nb_conv[1], 
                            init='he_uniform', border_mode = 'valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool[1], nb_pool[1])))
    model.add(SpatialDropout2D(dropout[1]))

    model.add(Convolution2D(nb_filters[2], nb_conv[2], nb_conv[2], 
                            init='he_uniform', border_mode = 'valid')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool[2], nb_pool[2])))
    model.add(SpatialDropout2D(dropout[2]))
    
    model.add(Flatten())

    model.add(Dense(nb_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(dropout[3]))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                  metrics = ['accuracy'])
    model.summary()

    return model

def dense_net_construct(hps, AE = False):
    layer_dims = hps['layer_dims']
    input_dim = hps['input_dim']

    model = Sequential()
    for i,dim in enumerate(layer_dims):
        if i == 0:
            model.add(Dense(dim, input_dim = input_dim, init = 'he_uniform'))
        else:
            model.add(Dense(dim, init = 'he_uniform'))
        model.add(Activation('relu'))
    if AE == True:
        model.add(Dense(input_dim, init = 'he_uniform'))
        model.compile(loss = 'mse', optimizer = 'adam', 
                  metrics = ['accuracy'])
        
    else:
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                  metrics = ['accuracy'])

    
    model.summary()

    return model




