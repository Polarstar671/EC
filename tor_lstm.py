
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

from keras.models import Sequential



def build_model(learn_params, nb_classes):
    input_length = learn_params["maxlen"]
    input_dim = learn_params["nb_features"]
    layers = learn_params["layers"]

    model = Sequential()


    if len(layers) == 0:
        raise ("No layers")

    if len(layers) == 1:
        layer = layers[0]
        model.add(LSTM(input_shape=(input_length, input_dim),

                       units=layer.as_int('units'),
                       activation=layer['activation'],
                       recurrent_activation=layer['rec_activation'],
                       return_sequences=False,

                       dropout=layer.as_float('dropout')))
        model.add(Dense(units=nb_classes, activation='softmax'))
        return model

    first_l = layers[0]
    last_l = layers[-1]
    middle_ls = layers[1:-1]

    model.add(LSTM(input_shape=(input_length, input_dim),

                   units=first_l.as_int('units'),
                   activation=first_l['activation'],
                   recurrent_activation=first_l['rec_activation'],
                   return_sequences=True,

                   dropout=first_l.as_float('dropout')))
    for l in middle_ls:
        model.add(LSTM(units=l.as_int('units'),
                       activation=l['activation'],
                       recurrent_activation=l['rec_activation'],
                       return_sequences=True,

                       dropout=l.as_float('dropout')))

    model.add(LSTM(units=last_l.as_int('units'),
                   activation=last_l['activation'],
                   recurrent_activation=last_l['rec_activation'],
                   return_sequences=False,

                   dropout=last_l.as_float('dropout')))

    model.add(Dense(units=nb_classes, activation='softmax'))
    return model

