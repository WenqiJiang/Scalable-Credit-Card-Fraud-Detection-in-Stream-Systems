import time
import numpy as np

from socket import socket, AF_INET, SOCK_STREAM

if __name__ == '__main__':
    serverPort = 7017
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
        
        # message = string_to_send.encode()
        # connectionSocket.send(message)
        # print ("server handled: " + str(addr) + " with message: ", str(message))
        # connectionSocket.close()

        # # change sleep time according to current feed_freq
        # if index != 0 and index % group_size == 0:
        #     feed_freq += feed_freq_increment
        #     sleep_time = 1.0/feed_freq

        # # sleep for calculated time period
        # # time.sleep(sleep_time)


        # initiate single batch data transmission
        if (index + 1) % batch_size == 0:
            connectionSocket, addr = serverSocket.accept()
            message = conca_string.encode()
            connectionSocket.send(message)

            # reset conca_string
            conca_string = '{} {} '.format(feed_freq,batch_size)

            print ("server handled: " + str(addr) + " with message: ", str(message))
            connectionSocket.close()

        # alter feed_freq
        # if (index + 1) % group_size == 0:
        #     feed_freq += feed_freq_increment
        #     sleep_time = 1.0/feed_freq

        #     ad-hoc test termination
        #     break

        # sleep
        # time.sleep(sleep_time)

        # ------------------------------
        # ad-hoc test termination
        # if index >= batch_size - 1:
        #     break
        # ------------------------------

    # ------------------------------------------------------------



    # message = str(np.random.randn(5)).encode()
    # # message = bytes([1])
    # for j in range(5):
    #     connectionSocket, addr = serverSocket.accept()
    #     connectionSocket.send(message)
    #     print ("server handled: " + str(addr) + " with message: ", str(message))

    #     connectionSocket.close()
    #     time.sleep(1)