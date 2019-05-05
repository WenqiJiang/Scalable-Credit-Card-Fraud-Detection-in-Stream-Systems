import time
import numpy as np

from socket import socket, AF_INET, SOCK_STREAM

if __name__ == '__main__':
    serverPort = 9994
    serverSocket = socket(AF_INET,SOCK_STREAM) ## LISTENING SOCKET!!!
    serverSocket.bind(('localhost',serverPort))
    serverSocket.listen(1)
    print ('The server is ready to receive')

    message = np.ones(3)
    for j in range(5):
        connectionSocket, addr = serverSocket.accept()
        connectionSocket.send(message)
        print ("server handled: " + str(addr) + " with message: ", str(message))

        connectionSocket.close()
        time.sleep(1)