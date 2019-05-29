import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from keras import layers, models, optimizers, regularizers
from sklearn.model_selection import StratifiedKFold, train_test_split

image_ext = ['npy']
coords = "../coords_new"
labels = "../labels.xlsx"
num_joints = 19
num_dimms = 3
seq_len = 80
seed = 7
num_classes = 3
lossfunct = 'categorical_crossentropy'
offset = 0

# Set random seed
np.random.seed(seed)

def clean_folder(path):
    for filename in os.listdir(path):
        if is_numpy(filename):
            num = filename[:-4]
            num = num.zfill(5)
            new_filename = num + ".npy"
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

def is_numpy(file_name):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext

def get_filename(file_name):
    name = file_name[0: + file_name.rfind('.')]
    return name

def ImportData():
    
    x_total = np.empty((0,seq_len,num_dimms*num_joints+offset))
    
    list = os.listdir(coords)
    
    for file in sorted(list):
        if is_numpy(file):
            x = np.load(os.path.join(coords,file))
            #print(x.shape)
            x_total = np.append(x_total, [x], axis=0)

    df = pd.read_excel(labels, index_col=None, na_values=['NA'], usecols = "B,K") #k for top, good, fat (label is "lab"), f for good, bad (label is "good")
    y_total = df['lab']

    y = np.empty(y_total.shape)
    
    y[y_total == 'g'] = 1
    y[y_total == 'f'] = 0
    y[y_total == 't'] = 2
    '''
    y[y_total != 'x'] = 0
    y[y_total == 'x'] = 1
    '''
    y = keras.utils.to_categorical(y, num_classes)
    x_train= x_total
    y_train= y
    x_test = np.empty((0))
    y_test = np.empty((0))
    x_train, x_test, y_train, y_test = train_test_split(x_total, y, test_size=0.1935, random_state=seed, stratify=y)
    data1 = namedtuple("data",'x_train y_train x_test y_test')
    params = data1(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return params


def RunNetwork(x_train,y_train, x_test, y_test):
    
    # Learning Parameters
    batch_size = num_classes * 8
    epochs = 600
    validation_split_nn = 0.24
    hidden_vect_len = 48
    hidden_vect_len2 = 64
    learning_rate = .003
    drop = 0.2
    lamb = 0.01
    
    # Conv1
    conv1_filters = 100
    conv1_kernel = 11
    conv2_filters = 160
    conv2_kernel = 11
    conv3_filters = 100
    conv3_kernel = 7
    conv1_padding = 'same'
    conv1_activation = 'relu'
    data_input_shape = (seq_len,num_dimms*num_joints+offset)

    
    # Pool1
    pool1_size = 3
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    lstm_1 = layers.LSTM(units=hidden_vect_len, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l1(lamb), recurrent_regularizer=regularizers.l1(lamb), bias_regularizer=regularizers.l1(lamb), activity_regularizer=regularizers.l1(lamb), kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=drop, recurrent_dropout=drop, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)
    
    #lstm = layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    
    lstm_2 = layers.LSTM(units=hidden_vect_len2, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l1(lamb), recurrent_regularizer=regularizers.l1(lamb), bias_regularizer=regularizers.l1(lamb), activity_regularizer=regularizers.l1(lamb), kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=drop, recurrent_dropout=drop, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)
    
    # Form model
    model = models.Sequential()
    model.add(layers.Bidirectional(lstm_1))
    model.add(layers.BatchNormalization())
    #    model.add(layers.Bidirectional(lstm_1))
    #    model.add(layers.Conv1D(conv1_filters, conv1_kernel, padding=conv1_padding, activation=conv1_activation, input_shape=data_input_shape))
    model.add(layers.Conv1D(conv1_filters, conv1_kernel, padding=conv1_padding, activation=conv1_activation, input_shape=data_input_shape))
    model.add(layers.MaxPool1D(pool1_size))
    model.add(layers.BatchNormalization())
    #    model.add(layers.Conv1D(conv2_filters, conv2_kernel, padding=conv1_padding, activation=conv1_activation, input_shape=data_input_shape))
    #    model.add(layers.Conv1D(conv2_filters, conv2_kernel, padding=conv1_padding, activation=conv1_activation, input_shape=data_input_shape))
    #    model.add(layers.MaxPool1D(pool1_size))
    #    model.add(layers.Conv1D(conv3_filters, conv3_kernel, padding=conv1_padding, activation=conv1_activation, input_shape=data_input_shape))
    #    model.add(layers.Conv1D(conv3_filters, conv3_kernel, padding=conv1_padding, activation=conv1_activation, input_shape=data_input_shape))
    #    model.add(layers.MaxPool1D(pool1_size))
    model.add(layers.Flatten())
    model.add(layers.Dropout(drop))
    #model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dropout(drop))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(drop))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(drop))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    adm = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(loss=lossfunct, optimizer=adm, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size, epochs,validation_split=validation_split_nn)
    score =  model.evaluate(x_test, y_test, verbose=2)
    ynew = model.predict_classes(x_test)
    for i in range(len(ynew)):
        # show the inputs and predicted outputs
        print("Predicted=%s, Real Label=%s " % (ynew[i], y_test[i]))
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    model.save('SkeletalActionRecognition.h5')
    print(model.summary())
    
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def main():
    params = ImportData()
    RunNetwork(**params._asdict())

if __name__ == '__main__':
    main()
