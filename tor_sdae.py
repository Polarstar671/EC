from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.models import Model
import keras.utils.np_utils as npu
from keras.datasets import mnist
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop

import matplotlib

matplotlib.use('Agg')



def minst_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = npu.to_categorical(y_train, 10)
    y_test = npu.to_categorical(y_test, 10)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return (x_train, y_train), (x_test, y_test)


def make_layer(layer, x_train, x_test, steps=0, gen=False):
    in_dim = layer.as_int('in_dim')
    out_dim = layer.as_int('out_dim')
    epochs = layer.as_int('epochs')
    batch_size = layer.as_int('batch_size')
    optimizer = layer['optimizer']
    enc_act = layer['enc_activation']
    dec_act = layer['dec_activation']

    if optimizer == "sgd":
        optimizer = SGD(lr=layer[optimizer].as_float('lr'),
                        decay=layer['sgd'].as_float('decay'),
                        momentum=layer['sgd'].as_float('momentum'))
                        #nesterov=layer['sgd']['nesterov'])
    elif optimizer == "adam":
        optimizer = Adam(lr=layer['adam'].as_float('lr'),
                         decay=layer['adam'].as_float('decay'))
    elif optimizer == "rmsprop":
        optimizer = RMSprop(lr=layer['rmsprop'].as_float('lr'),
                            decay=layer['rmsprop'].as_float('decay'))



    input_data = Input(shape=(in_dim,))

    encoded = Dense(out_dim, activation=enc_act)(input_data)

    decoded = Dense(in_dim, activation=dec_act)(encoded)


    autoencoder = Model(input_data, decoded)


    encoder = Model(input_data, encoded)

    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')


    if gen:
        (train_steps, test_steps) = steps
        autoencoder.fit_generator(x_train, steps_per_epoch=train_steps, epochs=epochs)
    else:
        autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)



    if gen:
        (train_steps, test_steps) = steps
        new_x_train1 = encoder.predict_generator(x_train, steps=train_steps)
        new_x_test1 = encoder.predict_generator(x_test, steps=test_steps)
    else:
        new_x_train1 = encoder.predict(x_train)
        new_x_test1 = encoder.predict(x_test)

    weights = encoder.layers[1].get_weights()

    return new_x_train1, new_x_test1, weights



def build_model(learn_params, nb_classes, train_gen, test_gen, steps=0, pre_train=True):
    layers = learn_params["layers"]


    input_data = Input(shape=(layers[0].as_int('in_dim'),))
    prev_layer = input_data

    i = 0
    encoded_layers = []
    for l in layers:
        encoded = Dense(l.as_int('out_dim'), activation=l['enc_activation'])(prev_layer)
        i += 1
        encoded_layers.append(i)
        dropout = l.as_float("dropout")
        if dropout > 0.0:
            drop = Dropout(dropout)(encoded)
            i += 1
            prev_layer = drop
        else:
            prev_layer = encoded

    softmax = Dense(nb_classes, activation='softmax')(prev_layer)
    sae = Model(input_data, softmax)

    if pre_train:

        prev_x_train = None
        prev_x_test = None
        for i, l in enumerate(layers):
            if i == 0:
                prev_x_train, prev_x_test, weights = make_layer(l, train_gen, test_gen, steps=steps, gen=True)
            else:
                prev_x_train, prev_x_test, weights = make_layer(l, prev_x_train, prev_x_test)
            sae.layers[encoded_layers[i]].set_weights(weights)


    return sae

