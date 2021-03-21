// Duino code for 74HC595N

// data
int SER = 2;
// shift register clk. This is SRCLK or SHCP in the lit; RCLK is instead for storing/latching in the register.
int CLK = 3;
// Latch, aka RCLK (also STCP) for 74hc595n
int LATCH = 4;
// shift register clear. "INV" means negation: low asserts clearing, high is 'not-clear'.
int SRCLR_INV = 5;


// See 'function table' for basic functionality: https://www.ti.com/lit/ds/symlink/sn74hc595.pdf
void Setup() {
  // put your setup code here, to run once:
  pinMode(SER, OUTPUT);
  pinMode(CLK, OUTPUT);
  pinMode(LATCH, OUTPUT);
  pinMode(SRCLR_INV, OUTPUT);

  // Unclear
  digitalWrite(SRCLR_INV, HIGH);
  // Latch low; output will not be shown while data propagate
  digitalWrite(LATCH, LOW);
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
  // TODO: should probably latch when this occurs, but docs say that simply asserting SRCLR is enough.
}


// The main method for writing data to output, bit by bit, as a single fixed-size state.
// Unlatches, writes data lsb-first, then latches.
// Note: Caller is responsible for initial state and/or clearing.
// @data: the number whose bits are written, lowest bit first.
// @numBits: the number of bits to write; up to sizeof(int)
void Write(int data, int numBits) {
  digitalWrite(LATCH, LOW);
  
  for(int i = 0; i < numBits; i++) {
    bool nextBit = ((num >> i) & 1) == 1;
    shiftOut(nextBit, nextBit);

    // dumb overflow guard
    if(i > 32) {
      return; 
    }
  }

  digitalWrite(LATCH, HIGH);
}


// Test each output by writing a bit exclusively to each position with a small delay between.
void test() {
for(int i = 0; i < 8; i++) {
    Write(1 << i, 8);
    delay(500);
  }
}


void shifty() {
  for(int i = 0; i < 256; i++) {
    Write(i, 16);
    delay(500);
  }
}


void loop() {
  test();
  //shifty();
  //Shutdown();
}


// Shifts out one bit to the hardcoded shift register outputs
void shiftOut(bool state) {
  digitalWrite(SER, state);
  tick();
}


// A rising edge of the clk: low -> high.
// Note: caller is responsible for initial state
void rise() {
  digitalWrite(CLK, 1);
  delayMicroseconds(1);
}


void fall() {
  digitalWrite(CLK, 0);
  delayMicroseconds(1);
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
