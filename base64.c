#include "stdio.h"
//#include "stdlib.h"
#include "string.h"
#include "unistd.h" //for io in test functions only
#include "sys/fcntl.h"

#define DBG 1
#define ERR 64 //some value not in the base64 char-value set of numbers 0-63
#define IS_LITTLE_ENDIAN 1


typedef unsigned char U8;
typedef unsigned short int U16;

// IMU data packet: must mirror that used by server/client or misalignment will occur...
typedef struct imuVector{
  short int accX;  //accelerometer vals
  short int accY;
  short int accZ;
  short int gyroRoll; //gyroscope vals
  short int gyroPitch;
  short int gyroYaw;
  short int gyroTemp;
  short int magX;  //magnetometer vals
  short int magY;
  short int magZ;
  short int bmpPa; //barometer / temp vals
  short int bmpTemp;
}IMU;


U8 g_decTable[256];
U8* g_encTable = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/*
 = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',\
                                'P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d',\
                                'e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',\
                                't','u','v','w','x','y','z','0','1','2','3','4','5','6','7',\
                                '8','9','+','/'};
*/

void InitDecodeTable(U8 decTable[256])
{
  int i, j;

  for(i = 0; i < 256; i++){
    decTable[i] = ERR;
  }

  //set the values
  for(i = 0; i < 62; i++){
    if(i < 26){ //set the uppercase vals
      decTable[ g_encTable[i] ] = g_encTable[i] - 'A';
    }
    else if(i < 52){  //set lowercase vals
      decTable[ g_encTable[i] ] = g_encTable[i] - 'a' + 26;
    }
    else{  //set the numerical vals
      decTable[ g_encTable[i] ] = g_encTable[i] - '0' + 52;
    }
  }

  decTable['+'] = 62;
  decTable['/'] = 63;
  decTable['='] = 0;

  for(i = 0; i < 256; i++){
    if(decTable[i] == ERR){
      printf("__ ");
    }
    else{
      printf("%2u ",(unsigned int)decTable[i]);
    }

    if(i % 16 == 15){
      printf("\n");
    }
  }
  printf("\n");
}

//
U16 swapBytes(U16 shrt)
{
  /*
  U16 temp;
  printf("swap in: %x\n",shrt);
  temp = ((shrt & 0x00FF) << 8) | ((shrt & 0xFF00) >> 8);
  printf("out:  %x\n",temp);
  return temp;
  */
  return ((shrt & 0x00FF) << 8) | ((shrt & 0xFF00) >> 8);
}

//
short int my_htons(short int shrt)
{
  #ifdef IS_LITTLE_ENDIAN
    return swapBytes((U16)shrt);
  #else
    return shrt;
  #endif
}

//
short int my_ntohs(short int shrt)
{
  #ifdef IS_LITTLE_ENDIAN
    return swapBytes((U16)shrt);
  #else
    return shrt;
  #endif
}



void putVector(const IMU* vector)
{
  printf("\nVector:\n");
  printf("Acl:  %5u %5u %5u\n",vector->accX, vector->accY, vector->accZ);
  printf("Gyr:  %5u %5u %5u %5u\n",vector->gyroRoll, vector->gyroPitch, vector->gyroYaw,vector->gyroTemp);
  printf("Mag:  %5u %5u %5u\n",vector->magX, vector->magY, vector->magZ);
  printf("Bar:  %5u %5u\n\n",vector->bmpPa, vector->bmpTemp);
}


void ntohsImuVector(IMU* vec)
{
  //my_ntohs all the vals
  vec->accX      = my_ntohs(vec->accX);
  vec->accY      = my_ntohs(vec->accY);
  vec->accZ      = my_ntohs(vec->accZ);
  vec->gyroRoll  = my_ntohs(vec->gyroRoll);
  vec->gyroPitch = my_ntohs(vec->gyroPitch);
  vec->gyroYaw   = my_ntohs(vec->gyroYaw);
  vec->gyroTemp  = my_ntohs(vec->gyroTemp);
  vec->magX      = my_ntohs(vec->magX);
  vec->magY      = my_ntohs(vec->magY);
  vec->magZ      = my_ntohs(vec->magZ);
  vec->bmpPa     = my_ntohs(vec->bmpPa);
  vec->bmpTemp   = my_ntohs(vec->bmpTemp);
}

void htonsImuVector(IMU* vec)
{
  //my_htons all the vals
  vec->accX      = my_htons(vec->accX);
  vec->accY      = my_htons(vec->accY);
  vec->accZ      = my_htons(vec->accZ);
  vec->gyroRoll  = my_htons(vec->gyroRoll);
  vec->gyroPitch = my_htons(vec->gyroPitch);
  vec->gyroYaw   = my_htons(vec->gyroYaw);
  vec->gyroTemp  = my_htons(vec->gyroTemp);
  vec->magX      = my_htons(vec->magX);
  vec->magY      = my_htons(vec->magY);
  vec->magZ      = my_htons(vec->magZ);
  vec->bmpPa     = my_htons(vec->bmpPa);
  vec->bmpTemp   = my_htons(vec->bmpTemp);
}

/*
  Decodes an imu vector from the input stream. Assume ibuf points to the beginning of
  an imu vector of 24 bytes.  Input will always be a multiple of four, but since
  IMU is 24 bytes, and both mods both 3 and 4, padding doesn't matter.
*/
void Deserialize(U8* ibuf, U8 obuf[], int nBytes)
{
  int i, j;
  U8 byte1, byte2, byte3, byte4;

  #ifdef DBG
  if(nBytes % 4 != 0){ //base64 encodings must necessarily be a multiple of 4
    printf("ERROR cannot base64-decode %d bytes, which is not a multiple of 4\n",nBytes);
    return;
  }
  #endif

  printf("In dec, nBytes: %d \n",nBytes);
  i = j = 0;
  do{
    byte1 = g_decTable[ ibuf[i]   ];
    byte2 = g_decTable[ ibuf[i+1] ];
    byte3 = g_decTable[ ibuf[i+2] ];
    byte4 = g_decTable[ ibuf[i+3] ];

    #if defined DBG
    if(byte1 == ERR){ //remove once confident in encoder
      printf("Error byte1 was invalid i=%d [i]=%uu %c %u\n",i,(unsigned int)ibuf[i],byte1,(unsigned int)byte1);
    }
    if(byte2 == ERR){ //remove once confident in encoder
      printf("Error byte2 was invalid i=%d [i]=%uu %c %u\n",i,(unsigned int)ibuf[i+1],byte2,(unsigned int)byte2);
    }
    if(byte3 == ERR){ //remove once confident in encoder
      printf("Error byte3 was invalid i=%d [i]=%uu %c %u\n",i,(unsigned int)ibuf[i+2],byte3,(unsigned int)byte3);
    }
    if(byte4 == ERR){ //remove once confident in encoder
      printf("Error byte4 was invalid i=%d [i]=%uu %c %u\n",i,(unsigned int)ibuf[i+3],byte4,(unsigned int)byte4);
    }
    #endif

    obuf[j]   = (byte1 << 2) | (byte2 >> 4);
    obuf[j+1] = (byte2 << 4) | (byte3 >> 2);
    obuf[j+2] = (byte3 << 6) | byte4;
    j += 3;
    i += 4;
  }while(i < nBytes);

  obuf[i-3] = '\0';

  printf("decode complete; i=%u nBytes=%u\n",i,nBytes);
}

/*
  Returns base-64 encoding of imu vector for putting on a wire.
  This function is for 32 bit MCUs only.
*/
void Serialize(U8* ibuf, U8 obuf[], int nBytes)
{
  int i, j, padBytes, lastSegment;
  U8 tabIndex;

  //printf("in enc, serializing %s\n",ibuf);
  lastSegment = nBytes - nBytes % 3;
  padBytes = 3 - nBytes % 3;
  if(padBytes == 3){
    padBytes = 0;
  }

  //printf("padbytes: %d  nBytes: %d  lastsegment: %d\n",padBytes,nBytes,lastSegment);
  for(i = 0, j = 0; i < lastSegment; i += 3, j += 4){
    //get the first output byte
    tabIndex = (ibuf[i] & 0xFC) >> 2;
    //printf("1 tabindex: %d\n",tabIndex);
    obuf[j] = g_encTable[tabIndex];
    //get second
    tabIndex = ((ibuf[i] & 0x03) << 4) | ((ibuf[i+1] & 0xF0) >> 4);
    //printf("2 tabindex: %d\n",tabIndex);
    obuf[j+1] = g_encTable[tabIndex];
    //get third
    tabIndex = ((ibuf[i+1] & 0x0F) << 2) | ((ibuf[i+2] & 0xC0) >> 6);
    //printf("3 tabindex: %d\n",tabIndex);
    obuf[j+2] = g_encTable[tabIndex];
    //get fourth
    tabIndex = ibuf[i+2] & 0x3F;
    //printf("4 tabindex: %d\n",tabIndex);
    obuf[j+3] = g_encTable[tabIndex];
  }

  //printf("post loop i=%d j=%d nseg=%d\n",i,j,lastSegment);
  if(padBytes > 0){
    //printf("padding %d chars\n ",padBytes);
    //get the first output byte, as normal
    tabIndex  = (ibuf[i] & 0xFC) >> 2;
    obuf[j] = g_encTable[tabIndex];
    //printf("ibuf[i]=%u\n",(U16)ibuf[i]);

    if(padBytes == 1){
      //get second, as normal
      tabIndex  = ((ibuf[i] & 0x03) << 4) | ((ibuf[i+1] & 0xF0) >> 4);
      obuf[j+1] = g_encTable[tabIndex];
      //get third
      tabIndex  = (ibuf[i+1] & 0x0F) << 2;
      obuf[j+2] = g_encTable[tabIndex];
      //fourth is pad
      obuf[j+3] = '=';
    }
    else if(padBytes == 2){
      //printf("padding\n");
      //get half of second
      tabIndex  = (ibuf[i] & 0x03) << 4;
      obuf[j+1] = g_encTable[tabIndex];
      //third and fourth are pad char
      obuf[j+2] = '=';
      obuf[j+3] = '=';
    }
    obuf[j+4] = '\0';
  }
  else{
    obuf[j] = '\0';
  }
}

void SerializeImuVector(IMU* vec, char obuf[])
{
  //htons the vector values
  htonsImuVector(vec);
  printf("vec after htons:\n");
  putVector(vec);
  //base64 serialize and place in obuf
  Serialize((char*)vec,obuf,sizeof(IMU));
}

// ibuf MUST BE NULL TERMINATED
void DeserializeImuVector(U8* ibuf,IMU* vec)
{
  //base64 serialize and place in obuf
  Deserialize(ibuf,(U8*)vec,strnlen(ibuf,64));
  //ntohs the vector values on egress
  ntohsImuVector(vec);
}

void TestImuSerialization(void)
{
  IMU vec1 = {13,45,234,3435,423,875,12445,9,0,1,8656,76};
  const IMU expected = {13,45,234,3435,423,875,12445,9,0,1,8656,76};
  IMU vec2 = {999,999,999,999,999,999,999,999,999,999,999,999};
  U8 buf[8192] = {'\0'};
  int equal;

  buf[24] = '$';
  buf[25] = '\0';

  printf("vec1 before serialization:\n");
  putVector(&vec1);
  printf("begin enc...\n");
  SerializeImuVector(&vec1,buf);
  printf("vec1 after serialization:\n");
  putVector(&vec1);
  printf("%s\n",buf);
  DeserializeImuVector(buf,&vec2);
  printf("deseralized vector:\n");
  putVector(&vec2);

  equal = vec2.accX == expected.accX;
  equal = equal && (vec2.accY      == expected.accY);
  equal = equal && (vec2.accZ      == expected.accZ);
  equal = equal && (vec2.gyroRoll  == expected.gyroRoll);
  equal = equal && (vec2.gyroPitch == expected.gyroPitch);
  equal = equal && (vec2.gyroYaw   == expected.gyroYaw);
  equal = equal && (vec2.gyroTemp  == expected.gyroTemp);
  equal = equal && (vec2.magX      == expected.magX);
  equal = equal && (vec2.magY      == expected.magY);
  equal = equal && (vec2.magZ      == expected.magZ);
  equal = equal && (vec2.bmpPa     == expected.bmpPa);
  equal = equal && (vec2.bmpTemp   == expected.bmpTemp);
  if(equal){
    printf("PASS  TestImuSerialization\n");
  }
  else{
    printf("FAIL  TestImuSerialization\n");
    printf("Examine testVec and result:\n");
    printf("Expected values:");
    putVector(&expected);
    printf("Result values:");
    putVector(&vec2);
  }


}


void TestSerialize(void)
{
  char buf[256] = "172137790000-=][poiuytrewqasdfghjkl;'/.,mnbvcxz"; 
  char obuf[256];
  char result[256];
  char* expectedSerial = "MTcyMTM3NzkwMDAwLT1dW3BvaXV5dHJld3Fhc2RmZ2hqa2w7Jy8uLG1uYnZjeHo=";
  char* expectedDeserial = "172137790000-=][poiuytrewqasdfghjkl;'/.,mnbvcxz";

  printf("Serializing >%s<\n",buf);
  Serialize(buf,obuf,strlen(buf));
  printf("Serialized: >%s<\n",obuf);
  if(strncmp(expectedSerial,obuf,256) != 0){
   printf("ERROR expected serialization %s\n\t\tbut received %s\n",expectedSerial,obuf);
  }
  else{
    printf("SUCCESS serialization successful");
  }
  Deserialize(obuf,result,strlen(obuf));
  if(strncmp(expectedDeserial,result,256) != 0){
   printf("ERROR expected deserialization %s\n\t\tbut received %s\n",expectedDeserial,result);
  }
  else{
    printf("SUCCESS deserialization successful");
  }
  printf("Deserialized: >%s<\n",result);
}

void TestKoala(void)
{
  int ifile, ofile, n;
  char ibuf[4096];
  char obuf[8192];
  char deserialized[8192];

  ifile = open("./koala.jpg",O_RDONLY);
  ofile = open("./codec.jpg",O_WRONLY);
  while(n = read(ifile,ibuf,4095)){
    ibuf[4095] = '\0';
    printf("n=%d input: %s\n",n,ibuf);
    Serialize(ibuf,obuf,n);
    printf("len=%u serialized: %s\n",(int)strnlen(obuf,8192),obuf);
    Deserialize(obuf,deserialized,strnlen(obuf,8192));
    printf("len=%u writing: %s\n",n,deserialized);
    write(ofile,deserialized,n);
  }
  printf("TestKoala() completed\n");

  close(ifile);
  close(ofile);
}



int main(void)
{


  InitDecodeTable(g_decTable);
  TestSerialize();
  TestImuSerialization();

  //encode and decode the kaola in this directory
  TestKoala();






  return 0;
}

/*
  U8 buf[8192] = "/9j/4AAQSkZJRgABAQAAAQABAAD/4QBqRXhpZgAASUkqAAgAAAADADEBAgAHAAAAMgAAADsBAgAH\
                    AAAAOQAAAJiCAgAhAAAAQAAAAAAAAABHb29nbGUAQ29yYmlzAMKpIENvcmJpcy4gIEFsbCBSaWdo\
                    dHMgUmVzZXJ2ZWQuAAD/2wCEAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQ\
                    DQ4RDgsLEBYQERMWFRUVDA8XGBYUFxUSGhgBAwQEBgUGCgYGCg8ODA0PDAwNEBAMDQ8PEAwNDAwP\
                    DwwPEA8PDAwMDQwPDA0PDg8NDQ8PDQwMDQwNDQ0PDwwNDf/AABEIADIAMgMBEQACEQEDEQH/xAAZ\
                    AAADAQEBAAAAAAAAAAAAAAAGBwgJBQT/xAAzEAABAwIFAgUCBAcBAAAAAAABAgMEBREABhIhMQcT\
                    CCIyQVEUYTNSYnEVIySRobHhFv/EABoBAAIDAQEAAAAAAAAAAAAAAAMEAQIFAAb/xAAoEQACAgIA\
                    BQMEAwAAAAAAAAAAAQIRAyEEEjFBkSJRoRNh8PFCgdH/2gAMAwEAAhEDEQA/AJg6fZgrH/vorT88\
                    SYxd7S3y4tuOVX3vbY7+6hseScZ9GhLpY/Oq3VSk5GcZgGLHWgAyH2pziZDUsN3cCQbBegFJUpBJ\
                    Sryi/wAFcbdIXU6Ww88GnUmfNdqVerc3VRwVNwkyHCGnvOodsq21ITYAagbi33sHKqGMLtMpPPHW\
                    nLr8YQqzlmmLaXYIZkxEuBdiNQ2JIISoHja97mxwBxT/AIrwMU+8n5BOmw+l2ZT9fIpCqUqSoLab\
                    gTlICQeNlakpJ908DcbEYVlw0JakvDJkm1p/H6PNm/J/TLpzT5lahTp9UdgNOLRClJQpBWhOo6lJ\
                    SNVri4AvbexGOfBwbW3XtrwUcpVuv6EhB8X+Zn4UdyMpDsZbaVNLYpDPbUkjYpvva3F98aahFape\
                    BTrsSnQnJ7lNo71ak0WU8y5dMZxt0MNKWm+4uEqUm49QFj+2L13KzkqoAaF01rfWnxFTMuy3XZNH\
                    iJTUKkO0QhTbavTa97ArIAB3At8YMpJK2AUXJ0jUHKPRGndOsrrpbrjTTqoXaEVCm2wVBAsS2m+4\
                    CQBufSdjhZq3bHo6SUQki5Zpr1JZV9AjuJdU2UrQANrhTgINuALnje3yT1Im3Yss05Mgqc7KZfaZ\
                    acUt0KFkttpB3uFX5F7qvutQsL4HQSyKuvOb6pSBOp0Ram6U/HXBkzE3WpoHSEXII0nfkpNipO3I\
                    xfGtgMsnRPETP9SpkRmG3miO2iOhLSULQ2FJCRYA3F77e++HOX7CXMXjlbLuboeVXosmK59U84VN\
                    a1EFYv5bA8J+EjYDCP1tDX0HehgeHfJFO6RQKnX8ywYrFcqkouxm5DvaS4W9SWkKWs2UpSy6qx2O\
                    m99sVjPmDxxON+426X/FM+znpiy9GiJGr6d5zWUL0EaVrF/ypN/c3PF8FX3OfQHqzOrlOvDnSlOM\
                    g95brSVN6yDYgC1lGwO5/UdzsYRz0zgO9R4ctt+G7OMaolKXA2gkOA3TZOrSNJWLi4JTfSSFXviK\
                    IckTj4iMqTankwRo0aM5HktFK0BoOt/zAsJdBtsq7fCkg7A7jFoKmCyO46IzZ6X19DKErap6lhIC\
                    lKZWok/uDv8AvhzmQlRtBnHovOr1Sh1CgVtOYsrsuFbjYWFSGT6rEj8QWIF+QPb3x5riMMq5sbtf\
                    KPS8LlhzKOX0v4YofFe9lbM8WjUabU6lBFIqEedIZpS9Lk1Ia09pfyyQspNrbgn5xqYYuKSXsJ5M\
                    icm5d2/G9HK63ZyoU6pijVh16Vl/QlS6SystsSipAUe+m41p0kWbPl5JCiRY8Y7sVyZLFNTeoE3J\
                    Lk9FKJGV5Y78OE8tX9C9qsot/DagSdI4Unj4tKICOR7BrMeeKjkfNFHWmkqr1TqroQ1JYjl76Za1\
                    JASvzCwBTdShY/BTbeFW9hlC0m+/X8/0bNRpH8WYkU2e0lbiQppxQKbaQCth1AG6d1q+LWI++JrY\
                    Fy1QlpeVaz9U920uIb1nSkSAABfYenFLZFD/AOhldzxlFbL8aRKXSQ6lb7hV+HcgEk+w49jxjNnK\
                    to0sNy01aC+qT6dM6yUjM9VY+qpXaMqVFjgXeIFkptwUpsv3/Nbk4dwz5lYrmjyypGW3WPrFnPNf\
                    U6tVyqtvU1yozHZiIQ/DQlayrSkjawvbY+2HoJNC2S09oevhPp1U6103MBrEhSYUWOmOH1eUMAqC\
                    woXG51JSbfptiJLsU2ENAj1Pp5n2RQIkhUxRd1N1Iyz23UlVu6AQpIUkC5CSk23A2xTSZKbfUdEG\
                    rLeeBdS2qS4wlTqkualJATcIJtsfM5xtc/HNbJ6s58mmU2TIdeMhpJcUV2Efi5v+TAdBKG50VzjH\
                    guMMSHG1xZSdLybA3uOLjm98Y2WzYwNUM7rB0hi5jyW/UaLHMkQGygMNcvMKBS62i3NuQPsR74nh\
                    cjTafQjiYJpe+/Bn11p6I5SztIplYor6oT0ZoNTKehwhtmQLdxQbJOkLUNduDf5vjXeSlcTPVOlL\
                    qgLiz5OVX6XlbJ1Zf0tTBKriwsLbfSBYNOe1tvSCAPi5xXn1bKzpvR26jmtEqvuKix48QhSEMOFv\
                    +WSrVq3PA2tf/mIUrB8oSZj6pQsj01Ls57U8GyQ01ut832Sm52TYeo8AHnYYIrlpFehOFRzjmGr1\
                    CVOclVVDkp1T6kxlHtAqJJCLp9O+32wxSBlW+GufJfyBlxbsh1xZYYUVLWSSbDfGPnXqZqYjQXKT\
                    7hyA1dxRvFbJ3+2/+8Z0e5oy7EYeKKIwxm6c42y2245qC1IQAVbDk++NdJUYs36iRFgQWbRh9OC4\
                    kENeX2Pxjjj30h5xUGTdajeQkbn21Db/ACf74sR3FNmWU9NzlmhUh1x9TTyEtl1RUUCx2F+MaMV6\
                    UKPudxLLYSAG08fGLHH/2Q==";

*/




