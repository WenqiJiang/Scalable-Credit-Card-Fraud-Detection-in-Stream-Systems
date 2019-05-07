import time
import numpy as np
import argparse

from socket import socket, AF_INET, SOCK_STREAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="batch size, default 20")
    parser.add_argument('--port', type=int, default=6666, help="network port")
    args = parser.parse_args()
    batch_size = args.batch_size
    serverPort = args.port

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
    datastream = np.load("../data/origin_data/X_test.npy")
    print('load data complete')

    # initial concatenate str
    conca_string = '{} {} '.format(feed_freq,batch_size)

    # ------------------------------------------------------------
    # transaction transmission loop

    for index,data in enumerate(datastream):

        string_to_send = str(index) + ' ' + ' '.join(str(e) for e in data) + ' '
        conca_string += string_to_send

        # initiate single batch data transmission
        if (index + 1) % batch_size == 0:
            connectionSocket, addr = serverSocket.accept()
            message = conca_string.encode()
            connectionSocket.send(message)

            # reset conca_string
            conca_string = '{} {} '.format(feed_freq,batch_size)

            print ("server handled: " + str(addr) + " with message: ", str(message))
            connectionSocket.close()
