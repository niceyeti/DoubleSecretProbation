from socket import *

serverPort = 12000
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print "The server is ready to receive"

while 1:
  message, clientAddress = serverSocket.recvfrom(2048)
  modifiedMessage = message.upper()
  serverSocket.sendto(modifiedMessage, clientAddress)
  print "rx'ed: "+modifiedMessage


'''
from socket import *

serverPort = 12000
serverSocket = socket(AF_INET,SOCK_STREAM)
serverSocket.bind(('',serverPort))
serverSocket.listen(1)
print 'The server is ready to receive'

while 1:
  print "wating for server socket accept() ..."
  connectionSocket, addr = serverSocket.accept()
  print "accepted address: "+str(addr)
  sentence = connectionSocket.recv(1024)
  capitalizedSentence = sentence.upper()
  connectionSocket.send(capitalizedSentence)
  connectionSocket.close()
  print "Rx'ed: "+capitalizedSentence
'''
