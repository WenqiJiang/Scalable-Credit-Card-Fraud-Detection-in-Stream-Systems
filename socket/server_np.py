import time
import numpy as np

from socket import socket, AF_INET, SOCK_STREAM

if __name__ == '__main__':
    serverPort = 7015
    serverSocket = socket(AF_INET,SOCK_STREAM) ## LISTENING SOCKET!!!
    serverSocket.bind(('localhost',serverPort))
    serverSocket.listen(1)
    print ('The server is ready to receive')

    # set initial data feed rate (per second)
    feed_freq = 2
    sleep_time = 1.0/feed_freq

    # set increment rate and data group size
    feed_freq_increment = 1
    group_size = 100

    # load test data
    datastream = np.load("../data/subsamp_data/processed_X_test.npy")
    print('load data complete')

    # ------------------------------------------------------------
    # transaction transmission loop

    for index,data in enumerate(datastream):
        # single transaction transmission start

        # accept new connection
        connectionSocket, addr = serverSocket.accept()

        # construct raw string to send; delimit with space
        string_to_send = '{} {} '.format(feed_freq,index) + ' '.join(str(e) for e in data)
        message = string_to_send.encode()
        connectionSocket.send(message)
        print ("server handled: " + str(addr) + " with message: ", str(message))
        connectionSocket.close()

        # change sleep time according to current feed_freq
        if index != 0 and index % group_size == 0:
            feed_freq += feed_freq_increment
            sleep_time = 1.0/feed_freq

        # sleep for calculated time period
        time.sleep(sleep_time)
        

        # ------------------------------
        # ad-hoc test termination
        if index > 10:
            break
        # ------------------------------

        # single transaction transmission end

    # ------------------------------------------------------------

    # message = str(np.random.randn(5)).encode()
    # # message = bytes([1])
    # for j in range(5):
    #     connectionSocket, addr = serverSocket.accept()
    #     connectionSocket.send(message)
    #     print ("server handled: " + str(addr) + " with message: ", str(message))

    #     connectionSocket.close()
    #     time.sleep(1)