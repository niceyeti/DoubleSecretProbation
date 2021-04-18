
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

// Sets up the passed pins as outputs and configures their initial states.
void SetupRegisterPorts(int ser, int clk, int latch, int srclr_inv) {
  // put your setup code here, to run once:
  pinMode(ser, OUTPUT);
  pinMode(clk, OUTPUT);
  pinMode(latch, OUTPUT);
  pinMode(srclr_inv, OUTPUT);  
}

void InitializePorts(int ser, int clk, int latch, int srclr_inv)
{
  digitalWrite(ser, LOW);
  digitalWrite(clk, LOW);
  digitalWrite(latch, LOW); // latching low; data will propagate but not be shown
  digitalWrite(srclr_inv, HIGH); // unclear
}

// Runs once
void setup() {
  SetupRegisterPorts(SER, CLK, LATCH, SRCLR_INV);
  InitializePorts(SER, CLK, LATCH, SRCLR_INV);
}


// For convenience; manually upload this code instead to drive all ports low, before disconnecting arduino.
void shutdown(){
  digitalWrite(SER, false);
  digitalWrite(CLK, false);
  digitalWrite(LATCH, false);
}

// Shifts out the passed bit to the hardcoded shift register outputs
void shiftOut(bool state, bool withLatch){
  digitalWrite(SER, state);
  tick(withLatch);
}

// Ticks the clock once. If latch is true, then latch is clocked too, opposite of the clock.
void tick(bool withLatch){
  // Clock once (clk and latch, to opposite values; see timing diagram in SN74HC595N for an explanation)
  // tic
  digitalWrite(CLK, LOW);
  if(withLatch){
    digitalWrite(LATCH, HIGH);
  }
  delayMicroseconds(CLK_EDGE_US);
  
  // toc
  digitalWrite(CLK, HIGH);
  if(withLatch){
    digitalWrite(LATCH, LOW);
  }
  delayMicroseconds(CLK_EDGE_US);
}

void tickLatch(){
  digitalWrite(LATCH, HIGH);
  delayMicroseconds(CLK_EDGE_US);
  digitalWrite(LATCH, LOW);
}

// Clears register state with ~SRCLR: sets ~SRCLR to low, ticks, and then unclears.
void ClearOutput() {
  tick(true);
  digitalWrite(SRCLR_INV, LOW);
  tick(true);
  digitalWrite(SRCLR_INV, HIGH);
}

// @num: the number whose bits are written, lowest bit first.
// @numBits: the number of bits to write; up to sizeof(int)
void Write(unsigned int data, int numBits, bool lsbFirst) {
  bool nextBit;
  digitalWrite(LATCH, LOW);
  
  for(int i = 0; i < numBits; i++) {
    if(lsbFirst){
      nextBit = (1 << i) & data;
    }
    else{
      nextBit = (data >> (numBits - i - 1)) & 1;
    }
    
    shiftOut(nextBit, false);
    // dumb guard
    if(i > 32) {
      return;
    }
  }
  
  digitalWrite(LATCH, HIGH);
}



/////////////////////////////////////////////////////////////////////////////////////////////
// LED MATRIX REGISTER CODE   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


/*
The logical vs. physical pin mapping for the 788BS is not intuitive, and requires random mapping.
The scheme here is to write a 16 bit state to a register, representing the total matrix state.
This means that the 16 bit value can be the output of the mapping from logical row/col order
to the unusual physical pinning of the 788BS.
*/

// The indices 1-8 represent the logical row or col; the value stored at each 
// index is the physical pin on the device (1-16). These values are from the datasheet, aka some
// random pic on google images :P Note that the rows are the anode (+) and columns are 
// cathode (-). So to drive one led, the row is + and col is -.
// NOTE: pins numbering starts at 1, so be careful when bit shifting.
const unsigned int ROW_MAPPING[8] = { 9, 14, 8, 12, 1,  7,  2,  5};
const unsigned int COL_MAPPING[8] = {13,  3, 4, 10, 6, 11, 15, 16};

// Write a single led in an 8x8 788BS led matrix:
// 
// @row: The row to write; must be in [0,7]
// @col: The col to write; must be in [0,7]
void writeLed(int row, int col, bool state) {  
    // A 16 bit ushort is used to drive two daisy-chained 8bit shift registers;
    // the high 8 bits represent the row and the low 8 bits the column.

    // TODO: depending on the anode/cathode orientation of row vs col, one of these 1s will depend on @state.
    unsigned short rowBits = HIGH << row;
    unsigned short colBits = (state ? HIGH : LOW) << col;
    unsigned short data = (rowBits << 8) | colBits;

    Write((int)data, 16, false);
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

    Write((int)data, 16, false);
}


// Writes the passed state to a 8x8 led matrix.
// The state is written one pixel in the matrix at a time, with a small delay to illuminate that led.
// @state: a bit matrix in [row][col] order.
void WriteMatrix(bool state[8][8]) {
  unsigned int data = 0;

  Write(0xFFFF, 16, false);
  //Write(0x0, 16, false);

  // iterating the logical row and column as viewed from above
  for(int row = 0; row < 8; row++) {
        //Write(0x0, 16);
        
        for(int col = 0; col < 8; col++) {
          // map from the logical row and column to physical pins
          int row_pin = ROW_MAPPING[row] - 1;
          int col_pin = COL_MAPPING[col] - 1;
          
          if(state[row][col]) {            
            // Set the rows bits: set only the row bit
            data = 1 << row_pin;
            for(int i = 0; i < 8; i++) {
              data |= (1 << (COL_MAPPING[i] - 1));
            }
            data = data & ( ~(1 << col_pin));
            //unsigned int colBits = ~(HIGH << col_pin);
            //data = rowBits | colBits;
          }
          else {
            data = 0xFFFF;
          }
          
          Write(data, 16, false);

          if(state[row][col]) {
            delayMicroseconds(10);
          }
          else {
            delayMicroseconds(2);
          }
        }
        //Write(0xFFFF << 16, 16, false);
        Write(0xFFFF, 16, false);
    }
    
    // Write((int)data, 16);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// END OF LED MATRIX CODE     ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////////////////////////////////////////////
// MAIN APP CODE              ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


// Test each output by writing a bit exclusively to each position with a small delay between.
void test() {
  //Write(7, 8);

/*
 bool matrix[8][8] = {
 {true, true, true, true, true, true, true, true},
 {false, false, false, false, false, false, false, false},
 {true, true, true, true, true, true, true, true},
 {false, false, false, false, false, false, false, false},
 {false, false, false, false, false, false, false, false},
 {true, true, true, true, true, true, true, true},
 {false, false, false, false, false, false, false, false},
 {true, true, true, true, true, true, true, true}};
  */
 bool matrix[8][8] = {
 {true,  true, false, false, false, false, false, false},
 {false, true, false, false, false, false, false, false},
 {false, false, true, false, false, false, false, false},
 {false, false, false, true, false, false, false, false},
 {false, false, false, false, true, false, false, false},
 {false, false, false, false, false, true, false, false},
 {false, false, false, false, false, false, true, false},
 {false, false, false, false, false, false, false, true}};
  
  
  WriteMatrix(matrix);  

  /*  
  for(int i = 0; i < 256; i++) {
    //Write(i, 8);
    //Write(i, 16);
    WriteMatrix(matrix);
    delay(1000);
    //ClearOutput();
    //delay(1000);
  }
  */
}

void loop() {
  test();
  //Write((int)(0b1111111111111111 << 16), 16, true);  ClearOutput();
  //Write((unsigned int)(0b1100), 4, false);
  //Write((unsigned int)(0b1111111111111111 << 16), 16, true);
  //Write((int)((~0b1111111111101111) << 16), 16, false);
    //Write((int)((~0b11111111) << 16), 16, false);
    //Write((int)((~0b)), 16, true);
    //delay(1000);
    //Write(0, 16, true);
    //    delay(100);
  //ClearOutput();
  //    delay(1000);
}


