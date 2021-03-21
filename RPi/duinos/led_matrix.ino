// Duino code for 74HC595N

/*************************************************************************************************************
A single code file project for controlling the 8x8 led matrix, 788BS, using two daisy-chained 74HC595N
shift registers. This code is a mess because I'm not famliar with duino project organization, so this file
mixes library code and main()/application code, which needs to be refactored.

TODO:
- organize devices into libraries in a generalizable way
- break up this file

*************************************************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////
// 74HC595N SHIFT REGISTER CODE /////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

/*
TODO: The shift register code belongs in a library. Need to familiarize myself with duino
project organization.
*/

// TODO: these are main application constants; move them when librarifying.
// Disentangling these will be hard, unless there is oop for some kind of di to inject them.
// Compile time mechanisms from mcu-programming, like static vars, seem ugly. Hm...

// data
const int SER = 2;
// shift register clk. This is SRCLK or SHCP in the lit; RCLK is instead for storing/latching in the register.
const int CLK = 3;
// Latch, aka RCLK (also STCP) for 74hc595n
const int LATCH = 4;
// shift register clear. "INV" means negation: low asserts clearing, high is 'not-clear'.
const int SRCLR_INV = 5;
// Minmum clk edge duration. ~100 ns minimum according to the datasheet, 1000 ns oughta do it
const int CLK_EDGE_US = 1;


// Sets up the past pins as outputs and configures their initial states.
void SetupRegisterPorts(int ser, int clk, int latch, int srclr_inv) {
  // put your setup code here, to run once:
  pinMode(ser, OUTPUT);
  digitalWrite(ser, LOW);

  pinMode(clk, OUTPUT);
  digitalWrite(clk, LOW);
  
  pinMode(latch, OUTPUT);
  digitalWrite(latch, LOW); // latching low; data will propagate but not be shown
  
  pinMode(srclr_inv, OUTPUT);
  digitalWrite(srclr_inv, HIGH); // unclear  
}


// Forn convenience; manually upload this code instead to drive all ports low, before disconnecting arduino.
void Shutdown() {
  ClearOutput();

  digitalWrite(SER, LOW);
  digitalWrite(CLK, LOW);
  digitalWrite(LATCH, LOW);
  digitalWrite(SRCLR_INV, LOW);
}


// Clears register state with ~SRCLR: sets ~SRCLR to low, ticks, and then unclears.
void ClearOutput() {
  digitalWrite(SRCLR_INV, LOW);
  tick();
  digitalWrite(SRCLR_INV, HIGH);
}


// Shifts out one bit to the hardcoded shift register outputs
void shiftOut(bool state) {
  digitalWrite(SER, state);
  tick();
}

// The main method for writing data to output, bit by bit, as a single fixed-size state.
// Unlatches, writes data lsb-first, then latches.
// Note: Caller is responsible for initial state and/or clearing.
// @data: the number whose bits are written, lowest bit first.
// @numBits: the number of bits to write; up to sizeof(int)
void Write(int data, int numBits) {
  digitalWrite(LATCH, LOW);
  
  for(int i = 0; i < numBits; i++) {
    bool nextBit = ((data >> i) & 1) == 1;
    shiftOut(nextBit);

    // dumb overflow guard
    if(i > 32) {
      return; 
    }
  }

  digitalWrite(LATCH, HIGH);
}

// A rising edge of the clk: low -> high.
// Note: caller is responsible for initial state
void rise() {
  digitalWrite(CLK, 1);
  delayMicroseconds(CLK_EDGE_US);
}


void fall() {
  digitalWrite(CLK, 0);
  delayMicroseconds(CLK_EDGE_US);
}


// A tick is       ___
//            ____|   |____
//
// Note: caller is responsible for pre/post states.
void tick() {
  fall();  // 0
  rise();  // 1
  fall();  // 0
}


// A tock is _____     ______
//                |___|
//
// Note: caller is responsible for pre/post states.
void tock() {
  rise();  // 1
  fall();  // 0
  rise();  // 1
}

/////////////////////////////////////////////////////////////////////////////////////////////
// END OF SHIFT REGISTER CODE ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////
// LED MATRIX REGISTER CODE   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// Write a single led in an 8x8 788BS led matrix:
// 
// @row: The row to write; must be in [0,7]
// @col: The col to write; must be in [0,7]
void writeLed(int row, int col, bool state) {
    // A 16 bit ushort is used to drive two daisy-chained 8bit shift registers;
    // the high 8 bits represent the row and the low 8 bits the column.

    // TODO: depending on the anode/cathode orientation of row vs col, one of these 1s will depend on @state.
    unsigned short rowBits = 1 << row;
    unsigned short colBits = (state ? 1 : 0) << col;
    unsigned short data = (rowBits << 8) | colBits;

    Write((int)data, 16);
}


// Worth trying: write an entire row or column. The 0th entry in state array
// will be the highest bit in the row bits (the upper 8 bits of a ushort) to
// match left-to-right iteration.
void writeLedRow(int row, int col, bool rowState[8]) {
    // A 16 bit ushort is used to drive two daisy-chained 8bit shift registers;
    // the high 8 bits represent the row and the low 8 bits the column.
    unsigned short rowBits = 0;
    for(int i = 7; i >= 0; i--) {
        rowBits |= ((rowState[i] ? HIGH : LOW) << i);
    }

    unsigned short colBits = HIGH << col;

    unsigned short data = (rowBits << 8) | colBits;

    Write((int)data, 16);
}


// Writes the passed state to a 8x8 led matrix.
void WriteMatrix(bool state[8][8]) {
    for(int row = 0; row < 8; row++) {
        for(int col = 0; col < 8; col++) {
            writeLed(row, col, state[row][col]);
            delay(50);
            // TODO: shut off after writing
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// END OF LED MATRIX CODE     ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////
// MAIN APP CODE              ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


// See 'function table' for basic functionality: https://www.ti.com/lit/ds/symlink/sn74hc595.pdf
void setup() {
  SetupRegisterPorts(SER, CLK, LATCH, SRCLR_INV);
  
  /*
  // put your setup code here, to run once:
  pinMode(SER, OUTPUT);
  pinMode(CLK, OUTPUT);
  pinMode(LATCH, OUTPUT);
  pinMode(SRCLR_INV, OUTPUT);

  // Unclear
  digitalWrite(SRCLR_INV, HIGH);
  // Latch low; output will not be shown while data propagate
  digitalWrite(LATCH, LOW);
  */
}

// Test each output by writing a bit exclusively to each position with a small delay between.
void test() {
  //Write(7, 8);
  
  for(int i = 0; i < 256; i++) {
    //Write(i, 8);
    Write(i, 16);
    delay(100);
    //ClearOutput();
  }
}


void loop() {
  test();
}



