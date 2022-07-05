import argparse
import time
import os
from datetime import datetime
import random

import numpy as np
import math
from configobj import ConfigObj
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import model_from_json
from data import load_data, load_data2, split_dataset, DataGenerator
import tor_lstm
import tor_sdae
import tor_cnn


torconf = "tor.conf"
config = ConfigObj(torconf)
logfile = config['log']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def failRawData(rawData, failNum, noiseType, flowSizeList, zeroCode=int(1514 / 32)):
    sampleSize = rawData.shape[0]
    packseq = rawData.shape[1]

    if noiseType == 0:
        for i in range(sampleSize):
            curFlowSize = flowSizeList[i]

            tempList = []
            for j in range(curFlowSize):
                tempList.append(rawData[i][j])

            for j in range(int(failNum * curFlowSize)):
                curSampleLocation = random.randint(0, curFlowSize - j - 1)
                tempList.pop(curSampleLocation)

            for j in range(packseq):
                if j < len(tempList):
                    rawData[i][j] = tempList[j]
                else:
                    rawData[i][j] = 0.
    elif noiseType == 1:
        for i in range(sampleSize):
            curFlowSize = flowSizeList[i]

            tempList = []
            for j in range(curFlowSize):
                tempList.append(rawData[i][j])

            curFlowIdx = [j for j in range(curFlowSize)]
            sampLocations = random.sample(curFlowIdx, int(failNum * curFlowSize))
            AList = sampLocations.copy()
            sampLocations.sort()
            rawData[i][sampLocations] = rawData[i][AList]
    return rawData


def numpy_printopts(float_precision=6):
    float_formatter = lambda x: "%.{}f".format(float_precision) % x
    np.set_printoptions(formatter={'float_kind': float_formatter})


def curtime():
    return datetime.utcnow().strftime('%d.%m %H:%M:%S')


def log(id, s, dnn=None):
    print("> {}".format(s))
    if dnn is not None:
        l = open(dnn + "_log.out", "a")
    else:
        l = open(logfile, "a")
    l.write("ID{} {}>\t{}\n".format(id, curtime(), s))
    l.close()


def log_config(id):
    l = open("log_configs.out", "a")
    l.write("\nID{} {}\n".format(id, datetime.utcnow().strftime('%d.%m')))
    l.close()


def gen_id():
    return datetime.utcnow().strftime('%d%m_%H%M%S')


def plot_acc(acc, title, val_acc=None, comment="", imgdir='imgdir'):
    plt.figure(figsize=(10, 4))
    plt.ylim(0, 1)
    plt.plot(acc, label="Training", color='red')
    if val_acc is not None:
        plt.plot(val_acc, label="Validation", color='blue')
    plt.title(title, y=1.08)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode='expand', borderaxespad=0.)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('{}/acc{}.pdf'.format(imgdir, comment))
    plt.close()


def plot_loss(loss, title, val_loss=None, comment="", imgdir='imgdir'):
    plt.figure(figsize=(10, 4))
    plt.plot(loss, label="Training", color='purple')
    if val_loss is not None:
        plt.ylim(0, 5)
        plt.plot(val_loss, label="Validation", color='green')
    plt.title(title, y=1.08)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode='expand', borderaxespad=0.)
    plt.yticks(np.arange(0, 5, 0.5))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('{}/loss{}.pdf'.format(imgdir, comment))
    plt.close()


def entropy(probs):
    e = 0.0
    for prob in probs:
        if prob == 0.0:
            continue
        e += prob * math.log(prob)
    return -e


def log_results(id, fname, predicted, nb_classes, dnn=None, labels=None, resdir='resdir'):
    r = open("{}/{}.csv".format(resdir, fname), "w")

    r.write("correct;label;predicted;predicted_prob;entropy")

    for cl in range(0, nb_classes):
        r.write(";prob_{}".format(cl))

    r.write("\n")

    class_result = np.argmax(predicted, axis=-1)

    acc = 0.0
    num = len(predicted)
    for res in range(0, num):
        predicted_label = int(class_result[res])
        prob = predicted[res][predicted_label]
        ent = entropy(predicted[res])

        if labels is not None:
            label = int(np.argmax(labels[res], axis=-1))
            correct = int(label == predicted_label)
            acc += correct
        else:
            label = "-"
            correct = "-"
        r.write("{};{};{};{:.4f};{:.4f}".format(correct, label, predicted_label, prob, ent))
        for cl in range(0, nb_classes):
            r.write(";{:.4f}".format(predicted[res][cl]))
        r.write("\n")
    r.close()

    if labels is not None:
        acc /= num
        log(id, "Accuracy:\t{}".format(acc), dnn)

    log(id, "Predictions saved to {}".format(fname), dnn)


def predict(id, model, data, batch_size=1, steps=0, gen=False):
    if gen:
        score = model.evaluate_generator(data, steps)
        predicted = model.predict_generator(data, steps)
    else:
        (x, y) = data
        score = model.evaluate(x, y, batch_size=batch_size, verbose=1)
        predicted = model.predict(x)

    test_loss = round(score[0], 4)
    test_acc = round(score[1], 4)
    log(id, "Test loss(entropy):\t{}".format(test_loss))
    log(id, "Test accuracy:\t{}".format(test_acc))

    return predicted, test_acc, test_loss


def run(id, cv, data_params, learn_params, model=None):


    nb_classes = data_params["nb_classes"]


    print('Building model...')

    if model is None:
        if learn_params['dnn_type'] == "lstm":
            model = tor_lstm.build_model(learn_params, nb_classes)
        elif learn_params['dnn_type'] == "sdae":
            model = tor_sdae.build_model(learn_params, nb_classes,
                                         data_params['train_gen'], data_params['val_gen'],
                                         steps=(learn_params['train_steps'], learn_params['val_steps']),
                                         pre_train=False)
        else:
            model = tor_cnn.build_model(learn_params, nb_classes)

    metrics = ['accuracy']


    if learn_params['optimizer'] == "sgd":
        optimizer = SGD(lr=learn_params['lr'],
                        decay=learn_params['decay'],
                        momentum=0.9,
                        nesterov=True)
    elif learn_params['optimizer'] == "adam":
        optimizer = Adam(lr=learn_params['lr'],
                         decay=learn_params['decay'])
    else:
        optimizer = RMSprop(lr=learn_params['lr'],
                            decay=learn_params['decay'])

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    print(model.summary())

    start = time.time()

    history = model.fit_generator(generator=data_params['train_gen'],
                                  steps_per_epoch=learn_params['train_steps'],
                                  validation_data=data_params['val_gen'],
                                  validation_steps=learn_params['val_steps'],
                                  epochs=learn_params['epochs'])

    log(id, 'Training took {:.2f} sec'.format(time.time() - start))


    tr_loss = round(history.history['loss'][-1], 4)

    return tr_loss, model


def parse_model_name(model_path):
    name = os.path.basename(model_path)
    return name.split("_")[0] + "_" + name.split("_")[1], name.split("_")[2]


def load_model(model_path):

    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(model_path + ".h5")
    return loaded_model



def main(save=False, wtime=False):
    id = gen_id()

    datapath = config['datapath']
    cross_val = config.as_int('cv')
    traces = config.as_int('traces')
    dnn = config['dnn']
    seed = config.as_int('seed')
    minlen = config.as_int('minlen')

    nb_epochs = config[dnn].as_int('nb_epochs')
    batch_size = config[dnn].as_int('batch_size')
    val_split = config[dnn].as_float('val_split')
    test_split = config[dnn].as_float('test_split')
    optimizer = config[dnn]['optimizer']
    nb_layers = config[dnn].as_int('nb_layers')
    layers = [config[dnn][str(x)] for x in range(1, nb_layers + 1)]
    lr = config[dnn][optimizer].as_float('lr')
    decay = config[dnn][optimizer].as_float('decay')
    maxlen = config[dnn].as_int('maxlen')


    nb_features = 1


    start = time.time()
    data, labels, flowSizeList = load_data2(datapath,
                             minlen=minlen,
                             maxlen=maxlen,
                             traces=traces,
                             dnn_type=dnn)

    data = failRawData(data, 0.8, 0, flowSizeList)


    end = time.time()

    print("Took {:.2f} sec to load.".format(end - start))

    nb_instances = data.shape[0]
    nb_cells = data.shape[1]
    nb_classes = labels.shape[1]
    nb_traces = int(nb_instances / nb_classes)

    log(id, 'Loaded data {} instances for {} classes: '
            '{} traces per class, {} Tor cells per trace'.format(nb_instances,
                                                                 nb_classes,
                                                                 nb_traces,
                                                                 nb_cells))


    log_exp_name = "experiments.csv"
    if os.path.isfile(log_exp_name):
        log_exp = open(log_exp_name, "a")
    else:
        log_exp = open(log_exp_name, "a")
        log_exp.write(";".join(["ID", "w", "tr", "tr length", "DNN", "N layers", "lr",
                                "epochs", "tr loss", "tr acc", "cv", "test loss", "test acc", "std acc"]) + "\n")

    all_test_acc = []
    all_test_loss = []
    best_model = None

    ID = id
    indices = np.arange(nb_instances)
    for cv in range(1, cross_val + 1):
        model = None



        np.random.shuffle(indices)
        num = nb_instances

        split = int(num * (1 - test_split))
        ind_test = np.array(indices[split:])

        num = indices.shape[0] - ind_test.shape[0]
        split = int(num * (1 - val_split))

        ind_val = np.array(indices[split:num])
        ind_train = np.array(indices[:split])

        start = time.time()
        train_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_train)
        val_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_val)

        end = time.time()
        print("Took {:.2f} sec to make the generators.".format(end - start))

        data_params = {'train_gen': train_gen,
                       'val_gen': val_gen,
                       # 'test_data': (x_test, y_test),
                       'nb_instances': nb_instances,
                       'nb_classes': nb_classes,
                       'nb_traces': nb_traces}

        learn_params = {'dnn_type': dnn,
                        'epochs': nb_epochs,
                        'train_steps': ind_train.shape[0] // batch_size,
                        'val_steps': ind_val.shape[0] // batch_size,
                        'nb_features': nb_features,
                        'batch_size': batch_size,
                        'optimizer': optimizer,
                        'nb_layers': nb_layers,
                        'layers': layers,
                        'lr': lr,
                        'decay': decay,
                        'maxlen': maxlen}

        log(id, "Experiment {}: seed {}".format(cv, seed))

        tr_loss, model = run(id, cv, data_params, learn_params, model)


        x_test = np.take(data, axis=0, indices=ind_test)
        y_test = np.take(labels, axis=0, indices=ind_test)

        start = time.time()
        predictions, test_acc, test_loss = predict(id, model, (x_test, y_test), batch_size)
        print(predictions.shape)
        log(id, 'Test took {:.2f} sec'.format(time.time() - start))

        print(y_test)
        print(y_test.shape)
        rightNum = 0
        for temp_idx in range(y_test.shape[0]):
            if np.argmax(predictions[temp_idx]) == np.argmax(y_test[temp_idx]):
                rightNum = rightNum + 1

        precision_array = np.zeros((predictions.shape[1]))
        recall_array = np.zeros((predictions.shape[1]))
        f1_array = np.zeros((predictions.shape[1]))
        for i in range(f1_array.shape[0]):
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            ans = 0
            for j in range(y_test.shape[0]):
                label = np.argmax(y_test[j])
                if label == i:
                    ans += 1
                pre = np.argmax(predictions[j])
                if label == i and pre == i:
                    TP += 1
                if label == i and pre != i:
                    FN += 1
                if label != i and pre == i:
                    FP += 1
                if label != i and pre != i:
                    TN += 1
            precision_array[i] = TP / (TP + FP + 0.01)
            recall_array[i] = TP / (TP + FN + 0.01)
            print(TP, FP, TN, FN, ans)

        cnt = 0
        for i in range(f1_array.shape[0]):
            if precision_array[i] + recall_array[i] == 0:
                f1_array[i] = 0
                continue
            cnt += 1
            f1_array[i] = (2 * precision_array[i] * recall_array[i]) / (precision_array[i] + recall_array[i])

        for i in range(f1_array.shape[0]):
            print(precision_array[i], recall_array[i], f1_array[i])

        print('accuracy :%.2f', rightNum / y_test.shape[0])
        print('average precision为:%.2f', np.sum(precision_array) / f1_array.shape[0])
        print('average recall为:%.2f', np.sum(recall_array) / f1_array.shape[0])
        print('average f1_score为:%.2f', np.sum(f1_array) / f1_array.shape[0])

        if cross_val == 1:
            best_model = model
            break

        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)


    id = ID



    if save:

        model_json = best_model.to_json()
        with open("models/{}_{}.json".format(id, dnn), "w") as json_file:
            json_file.write(model_json)

        best_model.save_weights("models/{}_{}.h5".format(id, dnn))
        print("Saved model {}_{} to disk".format(id, dnn))


        del best_model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

    parser.add_argument('--save', '-s',
                        action="store_true",
                        help='save the trained model (for cv: the best one)')
    parser.add_argument('--wtime', '-wt',
                        action="store_true",
                        help='time experiment: test time datasets')
    parser.add_argument('--eval', '-e',
                        action="store_true",
                        help='test the model')

    args = parser.parse_args()

    if not args.eval:
        main(args.save, args.wtime)
