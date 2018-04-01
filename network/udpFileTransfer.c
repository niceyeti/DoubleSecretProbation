/*
  The start for this came someone's homework question on the internet, but it seemed like a very useful exercise.
  http://cboard.cprogramming.com/networking-device-communication/135947-file-transfer-using-udp-ip.html

  My own goal is to implement a simple file transfer client/server set up. The usage will be
  client connects to server at specific port, server acknowledges. Client and server
  verify one another somehow. Then client sends request for a file in the cwd
  directory of the server.

  <both client and server running>
  1) Client connects to server
  2) Client and server verify one another
  3) Client sends name of file to request
  4) Server returns file, or "FILE NOT FOUND"

  *Each packet of the file contents shall be error-checked for correctness (since this is UDP)
  *Each packet shall be ordered (packets can arrive out of order)
  *Remember, protocols to achieve these constraints can take many forms. We could mimic FTP and open
   two channels between client and server, one for control, and one for data. Stuff like that.

  Clearly, TCP should probably be used. Further, a higher-level api or library should be used for file X-fer, eg, FTP.
  This is just an exercise, that covers the bases of UDP, and could be used for other interesting things.

  And who cares about portability. Writing for linux sys-commands instead of stdlibrary i/o.
*/

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
 
#define MSG "This is a test message!"
#define BUFMAX 512
 
//Command Arguments:
// [0] - send
// [1] - destination machine
// [2] - destination port number
// [3] - file to be sent
 
int main(int argc, char* argv[])
{
  int sk;
  char buf[BUFMAX];

  


  int k;
  for(k=0;k<BUFMAX;k++){
    printf("The buffer contains %d\n",(char *)buf[k]);
  }

  struct sockaddr_in remote;
  struct hostent *hp;

  sk = socket(AF_INET, SOCK_DGRAM, 0);

  remote.sin_family = AF_INET;


  hp = gethostbyname(argv[1]);

  if(hp==NULL){
      printf("Can't find hostname. %s\n", argv[1]);
      exit(1);
  }

  bcopy(hp->h_addr, &remote.sin_addr.s_addr, hp->h_length);

  remote.sin_port = ntohs(atoi(argv[2]));

  dlsendto(sk, MSG, strlen(MSG)+1, 0, &remote,
  sizeof(remote),0);
  read(sk,buf,BUFMAX);
  printf("%s\n", buf);
  close(sk);
}
