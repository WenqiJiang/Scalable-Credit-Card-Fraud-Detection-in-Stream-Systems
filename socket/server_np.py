import time
import numpy as np
import argparse
import sys
from socket import socket, AF_INET, SOCK_STREAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="batch size, default 1")
    parser.add_argument('--port', type=int, default=6666, help="network port")
    parser.add_argument('--dataset', type=str, default="origin", help="choices: subsample / origin")
    args = parser.parse_args()
    batch_size = args.batch_size
    if batch_size >= 200:
        raise Exception("Socket Error, does not support batch size more than 100 currently")
    serverPort = args.port
    dataset = args.dataset

    serverSocket = socket(AF_INET,SOCK_STREAM) ## LISTENING SOCKET!!!
    serverSocket.bind(('localhost',serverPort))
    serverSocket.listen(1)
    print ('The server is ready to receive')

    # set initial data feed rate (per second)
    feed_freq = 1
    sleep_time = 1.0/feed_freq

    # set increment rate and data group size
    feed_freq_increment = 1
    group_size = 1000
    
    # set batch size (limit to max of 174)
    batch_size = 20

    # load test data
    # datastream = np.load("../data/subsamp_data/processed_X_test.npy")
    
    if dataset == "subsample":
        datastream = np.load("../data/subsamp_data/processed_X_test.npy")
        test_lable = np.load("../data/subsamp_data/processed_y_test.npy")
    if dataset == "origin":
        # raise Exception ("Almost all result of origin dataset are 0s, please use subsampled dataset for demo")
        datastream = np.load("../data/origin_data/X_test.npy")
        test_lable = np.load("../data/origin_data/y_test.npy")

    # initial concatenate str
    conca_string = '{} {} '.format(feed_freq,batch_size)

    # ------------------------------------------------------------
    # transaction transmission loop

    for index,data in enumerate(datastream):
        # print(index)

        string_to_send = str(index) + ' ' + ' '.join(str(e) for e in data) + ' '
        conca_string += string_to_send

        # initiate single batch data transmission
        if (index + 1) % 56000 == 0:
            print('string byte: ',sys.getsizeof(conca_string.encode()))
            print(sys.getsizeof(''.encode()))
            connectionSocket, addr = serverSocket.accept()
            message = conca_string.encode()
            connectionSocket.send(message)

            # reset conca_string
            conca_string = '{} {} '.format(feed_freq,batch_size)

            # print ("server handled: " + str(addr) + " with message: ", str(message))
            connectionSocket.close()
