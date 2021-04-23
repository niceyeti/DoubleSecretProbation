
// data
#define SER 2
// shift register clk. This is SRCLK or SHCP in the lit; RCLK is instead for storing/latching in the register.
#define CLK 3
// Latch, aka RCLK (also STCP) for 74hc595n
#define LATCH 4
// shift register clear. "INV" means negation: low asserts clearing, high is 'not-clear'.
#define SRCLR_INV 5
// Minmum clk edge duration. ~100 ns minimum according to the datasheet, 1000 ns oughta do it.
// NOTE: the min delay is 3 microseconds; below that, no guarantees.
#define CLK_EDGE_US 1
// The minimum possible delay on the arduino; 3 microseconds, per their docs.
#define MIN_DELAY_US 3

// The combined size of the shift register; in this case 2 shift registers, 8 bits each, 8*2 = 16.
#define NUM_BITS 16

// matrix size
#define NUM_ROWS 8
#define NUM_COLS 8

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
// LED MATRIX REGISTER CODE             ////////////////////////////////////////////////////
// 788BS common anode (row is +) matrix //////////////////////////////////////////////////////
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

// A type for expressing an led to turn on.
typedef struct MatrixLed {
  int Row;
  int Col;
};

// Writes the passed state to a 8x8 led matrix.
// The state is written one pixel in the matrix at a time, with a small delay to illuminate that led.
// @state: a bit matrix in [row][col] order.
void WriteMatrix(bool state[NUM_ROWS][NUM_COLS]) {
  unsigned int data = 0;

  //ClearMatrix();
  // iterating the logical row and column as viewed from above
  for(int row = 0; row < NUM_ROWS; row++) {
    for(int col = 0; col < NUM_COLS; col++) {
      // led at this row and col is on
      if(state[row][col]) {
        TurnOnLed(row, col);  
      }
      else {
        ClearMatrix();
      }
    }
  }
}

// Clear the matrix. This should be called to initialize state, or between states to clear previous.
void ClearMatrix() {
  /*
  // TODO: since this is a common-anode matrix the correct way to do this is to write 1s to all columns,
  // shutting them off, before writing 1s to the row pins. The following code works, but causes flashing output.
  unsigned int data = 0;
  for(int i = 0; i < 8; i++) {
    data |= (1 << (COL_MAPPING[i] - 1) );
  }
  Write(data, NUM_BITS, false);
  */
  // Write all 1's. Since this is a common-anode matrix, this shuts off all leds.
  Write(0xFFFF, NUM_BITS, false);
}

// Turn on a single led; all others are 'driven' to off. Since this is a common-anode matrix, driving the selected row high,
// and the selected col low, that led turns on; 
// This function is expected to be called repeatedly, such that the repetition creates the visual perception of a single matrix state.
// Note that row/col counting starts at 1, to match datasheet.
void TurnOnLed(int row, int col) {
  // map from the logical row and column to physical pins
  unsigned int row_pin = ROW_MAPPING[row] - 1;
  unsigned int col_pin = COL_MAPPING[col] - 1;
  
  // Build the next 16 bit state. Note there is some trixiness.
  // 1) Write 8 the row bits, such that only the selected row is high, since this code is for a common-anode matrix.
  //    This implies that the row bits only have one high bit.
  // 2) Write the column bits such that only the selected column is low, again since this is a common-anode matrix.
  //    This implies that the col bits only have one low bit.
  // This code is just the implementation of (1) and (2).
  unsigned int data = 1 << row_pin;
  for(int i = 0; i < NUM_COLS; i++) {
    data |= (1 << (COL_MAPPING[i] - 1));
  }
  data = data & ( ~(1 << col_pin) );
 
  // write the new state; written high-bit first, purely because of my wiring
  Write(data, NUM_BITS, false);
  // delay briefly to show output
  delayMicroseconds(MIN_DELAY_US);
}

// Turns on the passed leds.
// Note that row/col counting starts at 1, to match datasheet.
void TurnOnLeds(struct MatrixLed leds[], int n) {
  struct MatrixLed* led;
  for(int i = 0; i < n; i++) {
    led = &leds[i];
    TurnOnLed(led->Row, led->Col);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// END OF LED MATRIX CODE     ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////
// MAIN APP CODE              ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


// Test each output by writing a bit exclusively to each position with a small delay between.
void test_WriteMatrix() {
 bool matrix[NUM_ROWS][NUM_COLS] = {
 {true,  true, false, false, false, false, false, false},
 {false, true, false, false, false, false, false, false},
 {false, false, true, false, false, false, false, false},
 {false, false, false, true, false, false, false, false},
 {false, false, false, false, true, false, false, false},
 {false, false, false, false, false, true, false, false},
 {false, false, false, false, false, false, true, false},
 {false, false, false, false, false, false, false, true}};
  
  WriteMatrix(matrix);
}

void test_Animate() {
  for(int i = 0; i < NUM_ROWS; i++) {
    for(int j = 0; j < NUM_COLS; j++) {
      TurnOnLed(i, j);
      delay(75);
    }
  }
}


bool IsAlive(bool matrix[NUM_ROWS][NUM_COLS]) {
   for(int i = 0; i < NUM_ROWS; i++) {
     for(int j = 0; j < NUM_COLS; j++) {
       if(matrix[i][j]) {
         return true;
       }
     }
   }
  return false;
}

// TODO: could try wrapping the matrix boundaries, e.g. if one's rightmost neighbor is beyond the array bounds, then instead check the leftmost column, etc.
void test_Conway() {
 bool matrix[NUM_ROWS][NUM_COLS];
 bool next[NUM_ROWS][NUM_COLS];
 
 for(int i = 0; i < NUM_ROWS; i++) {
   for(int j = 0; j < NUM_COLS; j++) {
     matrix[i][j] = random(0,100000) % 2 == 0;
   }
 }

  int liveNeighbors = 0;
  int prevRow, prevCol, nextRow, nextCol;

  while(IsAlive(matrix)) {
    
    // update matrix per rules for conway's game of life
    for(int i = 0; i < NUM_ROWS; i++) {
      for(int j = 0; j < NUM_COLS; j++) {
        prevRow = i - 1;
        prevCol = j - 1;
        nextRow = i + 1;
        nextCol = j + 1;
        liveNeighbors = 0;
        
        if(prevRow >= 0) {
          // top-left
          if(prevCol >= 0 && matrix[prevRow][prevCol]) {
            liveNeighbors++;
          }
          // middle-top
          if(matrix[prevRow][j]) {
            liveNeighbors++;
          }
          // top-right
          if(nextCol < NUM_COLS && matrix[prevRow][nextCol]) {
            liveNeighbors++;
          }
        }

        // next-right        
        if(nextCol < NUM_COLS && matrix[i][nextCol]) {
          liveNeighbors++;
        }
        
        // prev-left
        if(prevCol > 0 && matrix[i][prevCol]) {
          liveNeighbors++;
        }
        
        if(nextRow < NUM_ROWS) {
          // bottom-left
          if(prevCol >= 0 && matrix[nextRow][prevCol]) {
            liveNeighbors++;
          }
          // middle-bottom
          if(matrix[nextRow][j]) {
            liveNeighbors++;
          }
          // bottom-right
          if(nextCol < NUM_COLS && matrix[nextRow][nextCol]) {
            liveNeighbors++;
          }
        }

        next[i][j] = (liveNeighbors == 2 && matrix[i][j]) || liveNeighbors == 3;
      }
    }
    
    // Copy the next state to the matrix
    for(int i = 0; i < NUM_ROWS; i++) {
      for(int j = 0; j < NUM_COLS; j++) {
        matrix[i][j] = next[i][j];
      }
    }
    
    for(int k = 0; k < 30; k++) {
      WriteMatrix(matrix);
      delayMicroseconds(3);
    }
  }
}

void loop() {
  test_Conway();
  //test_Animate();
  //test_WriteMatrix();
}


