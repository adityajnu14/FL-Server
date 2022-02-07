import numbers
import os
import grpc
import base64

import functions_pb2
import functions_pb2_grpc
import multiprocessing as mp
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import time
# import faulthandler
# faulthandler.enable()

path = "/home/aditya/Desktop/FL-Server/"

import warnings  
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf  
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense, Conv1D, Dropout, MaxPooling1D, Flatten
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical


#Utility function to convert model(h5) into string
def encode_file(file_name):
    with open('Models/'+file_name,'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string

clients_address = [ \
        '192.168.0.123:8081', \
        '192.168.0.112:8081',\
        '192.168.0.160:8081',\
        '192.168.0.144:8081', \
        '192.168.0.119:8081'\
         ]
n = len(clients_address)
stubs = []
for _ in range(n):
    stubs.append(functions_pb2_grpc.FederatedAppStub(\
        grpc.insecure_channel(clients_address[_])))


print("-"*40)
print('Total numbers of participating clients are {} '.format(n) )
print('-'*40)

def genFunc(i):
    empty = functions_pb2.Empty(value = 1)
    res = stubs[i].GenerateData(empty)
    print("client ",i+1,":",res.value)

def sendFunc(i, opt):
    if (opt == 2):
        filename = "InitModel.h5"
    else :
        filename = "optimised_model.h5"
    ModelString = functions_pb2.Model(model=encode_file(filename))
    res = stubs[i].SendModel(ModelString)
    print("client ",i+1,":",res.value, " - file :", filename)

def trainFunc(i):
    empty = functions_pb2.Empty(value = 1)
    res = stubs[i].Train(empty)
    with open("Models/model_"+str(i+1)+".h5","wb") as file:
        file.write(base64.b64decode(res.model))
    print("Saved model from client ",i)


def getTestDataset():

    database = sio.loadmat(path + 'Dataset/data_base_all_sequences_random.mat')
    
    X = database['Data_test_2']
    y = database['label_test_2']

    return X, y

def initializeServerMetrics():

    #Reseting globaal metrics file
    trainMetrics = {'accuracy' : [], 'loss' : []}
    with open('Models/globalMetrics.txt', "w") as f:
        f.write(json.dumps(trainMetrics))

def saveLearntMetrices(modelName):
     
    
    model = load_model(modelName)
    X_test, y_test = getTestDataset()
    y_test = to_categorical(y_test)
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Agreegated model on test data-> loss : {} and accuracy : {}".format(score[0], score[1]))

    with open('Models/globalMetrics.txt','r+') as f:
            trainMetrics = json.load(f)
            trainMetrics['accuracy'].append(score[1])
            trainMetrics['loss'].append(score[0])
            f.seek(0) 
            f.truncate()
            f.write(json.dumps(trainMetrics))

#The method split the data into number of devices
def createData():

    #Data preprocessing

    database = sio.loadmat(path + 'Dataset/data_base_all_sequences_random.mat')
    x_train = database['Data_train_2']
    y_train = database['label_train_2']
    #y_train_t = to_categorical(y_train)
    #x_train = (x_train.astype('float32') + 140) / 140 # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)
    #x_train2 = x_train[iii * samples:((iii + 1) * samples - 1), :] # DATA PARTITION

    # x_test = database['Data_test_2']
    # y_test = database['label_test_2']
    #x_test = (x_test.astype('float32') + 140) / 140
    #y_test_t = to_categorical(y_test)

    indices = database['permut']
    indices = indices - 1 # 1 is subtracted to make 0 at index
    indices = indices[0] # Open indexing

    i = 0
    slot = len(indices)//n
    data = []
    folderId = 0
    while i < len(indices) :
        if i//slot + 1 <= n:
            folderId = i//slot + 1
            data = []
        else:
            folderId = n

        for _ in range(slot):
            x = x_train[indices[i]]
            y = y_train[indices[i]]
            row = np.append(x, y)
            data.append(row)
            i = i + 1
            if(i == len(indices)):
                break

        df = pd.DataFrame(data)
        df.reset_index()
        outdir = path + 'Client data/client' + str(folderId)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        df.to_csv(outdir + '/data.csv')
    print("Dataset is created for %d devices" %(n))
  



""" These functions are called based on user input """

# Create local data for every participating device
def generateData():

    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(genFunc, i) for i in range(n)]
    concurrent.futures.wait(futures)

# Send latest model to all participating device 
def sendModel(opt):

    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(sendFunc, i, opt) for i in range(n)]
    concurrent.futures.wait(futures)
    
# Call for training for all participating devices
def train():

    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(trainFunc, i) for i in range(n)]
    concurrent.futures.wait(futures)


# This fucntion aggregates all models parmeters and create new optimized model 
def optimiseModels():

    print("inside optimize")
    models = [load_model("Models/model_"+str(i+1)+".h5") for i in range(n)]
    weights = [model.get_weights() for model in models]

    new_weights = list()

    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))

    new_model = models[0]
    new_model.set_weights(new_weights)
    new_model.save("Models/optimised_model.h5")
    print("Averaged over all models - optimised model saved!")
    saveLearntMetrices("Models/optimised_model.h5")

   
#Create and initilize model for first time. 
def createInitialModelForANN():
    K.clear_session()

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(512,)))
    model.add(Dense(16))
    # model.add(Dropout(0.2))
    model.add(Dense(8, activation='sigmoid'))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.save('Models/InitModel.h5')

def createInitialModelForCNN():
    K.clear_session()

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(512,)))
    # model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dense(filters=16, activation='relu'))

    model.add(Flatten())
    model.add(Dense(8, activation='sigmoid'))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.save('Models/InitModel.h5')



def visualizeTraining():

    fp =  open('Models/globalMetrics.txt','r')
    gloablMetrics = json.load(fp)

    f = plt.figure(1)
    plt.plot(gloablMetrics['accuracy'], label='Test')
    plt.title('Global model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig('Models/g_acc.png')
    f.show()
    
    g = plt.figure(2)
    plt.plot(gloablMetrics['loss'], label='Test')
    plt.title('Global loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig('Models/g_loss.png')
    g.show()


    h = plt.figure(3)
    for i in range(1, n+1):
        with open( '/home/aditya/Desktop/Training/CNN/Client' + str(i) + '/data/localMetrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['accuracy'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['accuracy'], '--b', label='Server')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig('Models/A_acc.png')
    h.show()

    s = plt.figure(4)
    for i in range(1, n+1):
        with open('/home/aditya/Desktop/Training/CNN/Client' + str(i) + '/data/localMetrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['loss'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['loss'], '--b', label='Server')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig('Models/A_loss.png')
    s.show()


if __name__ == '__main__':
    # mp.set_start_method('fork')
    round = 1
# User options for training main()
    while True:

        print("1. Initlize variables and creae data ")
        print("2. send initial model on participating clients")
        print("3. Start training on all clients")
        print("4. Aggregates all models")
        print("5. Send new model to all nodes")
        print("6. Visualize model accuracy/loss")
        print("7. Batch training on whole data")
        print("8. Exit")
        print("Enter an option: ")
        option = input()

        if (option == "1"):
            initializeServerMetrics()
            generateData() #data and index initilization at client level
            # createData()
        if (option == "2"):
            # createInitialModelForANN()
            createInitialModelForCNN
            saveLearntMetrices('Models/InitModel.h5')
            sendModel(int(option))
        if (option == "3"):
            train()
        if (option == "4"):
            optimiseModels()
        if (option == "5"):
            sendModel(int(option))
        if (option == "6"):
            visualizeTraining()
        if (option == "7"):
            for _ in range(10):
                print("Current Round ", round)
                train()
                # time.sleep(3)
                optimiseModels()
                sendModel(5)
                round = round +1
        if (option == "8"):
            break
        
                