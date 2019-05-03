# import socket

# HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
# PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print('Connected by', addr)
#         while True:
#             data = conn.recv(1024)
#             if not data:
#                 break
#             conn.sendall(data)


from socket import socket, AF_INET, SOCK_STREAM
import time

if __name__ == '__main__':
    serverPort = 9999
    serverSocket = socket(AF_INET,SOCK_STREAM) ## LISTENING SOCKET!!!
    serverSocket.bind(('',serverPort))
    serverSocket.listen(1)
    print ('The server is ready to receive')
    connectionSocket, addr = serverSocket.accept()

    while True:
        connectionSocket.send("adf adsf dfas".encode('ascii'))
        print ("server handled: " + str(addr) + " with message: " )
        # connectionSocket.close()
