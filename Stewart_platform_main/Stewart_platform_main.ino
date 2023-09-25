int receivedValues[6]; // Array to store the received integer values

#define LETTERS 6
#define DIGITS 5 

#define MIN_CONTROL_VALUE 50

#define M1A 44
#define M1B 42
#define M1PWM 7
#define M1Pot A0

#define M2A 40
#define M2B 38
#define M2PWM 6
#define M2Pot A1

#define M3A 36
#define M3B 34
#define M3PWM 5
#define M3Pot A2

#define M4A 32
#define M4B 30
#define M4PWM 4
#define M4Pot A3

#define M5A 28
#define M5B 26
#define M5PWM 3
#define M5Pot A4

#define M6A 24
#define M6B 22
#define M6PWM 2
#define M6Pot A5

int T = 0;
int ant = 0;
int desp = 0;

//Control para el motor 1
int M1Dir = 0; //0->STOP 1->FWRD 2-> BWRD
int M1Pos = 0;
float M1PosV = 0;
float M1PosR = 0;
float M1Error = 0;
float M1PastError = 0;
float M1DerError = 0;
float M1PastDerError = 0;
float M1W = 0.50;
float M1Control = 0;
float M1ControlValue = 0;
float M1Ref = 0;

float M1a = 0.0876;
float M1b = 1.6039;
float M1c = -0.21;

//Control para el motor 2
int M2Dir = 0; //0->STOP 1->FWRD 2-> BWRD
int M2Pos = 0;
float M2PosV = 0;
float M2PosR = 0;
float M2Error = 0;
float M2PastError = 0;
float M2DerError = 0;
float M2PastDerError = 0;
float M2W = 0.50;
float M2Control = 0;
float M2ControlValue = 0;
float M2Ref = 0;

float M2a = 0.06282;
float M2b = 1.7098;
float M2c = -0.12;

//Control para el motor 3
int M3Dir = 0; //0->STOP 1->FWRD 2-> BWRD
int M3Pos = 0;
float M3PosV = 0;
float M3PosR = 0;
float M3Error = 0;
float M3PastError = 0;
float M3DerError = 0;
float M3PastDerError = 0;
float M3W = 0.5;
float M3Control = 0;
float M3ControlValue = 0;
float M3Ref = 0;

float M3a = 0.06866;
float M3b = 1.70467;
float M3c = -0.24;

//Control para el motor 4
int M4Dir = 0; //0->STOP 1->FWRD 2-> BWRD
int M4Pos = 0;
float M4PosV = 0;
float M4PosR = 0;
float M4Error = 0;
float M4PastError = 0;
float M4DerError = 0;
float M4PastDerError = 0;
float M4W = 0.5;
float M4Control = 0;
float M4ControlValue = 0;
float M4Ref = 0;

float M4a = 0.0016;
float M4b = 1.9999;
float M4c = -0.14;

//Control para el motor 5
int M5Dir = 0; //0->STOP 1->FWRD 2-> BWRD
int M5Pos = 0;
float M5PosV = 0;
float M5PosR = 0;
float M5Error = 0;
float M5PastError = 0;
float M5DerError = 0;
float M5PastDerError = 0;
float M5W = 0.5;
float M5Control = 0;
float M5ControlValue = 0;
float M5Ref = 0;

float M5a = 0.07903;
float M5b = 1.6548;
float M5c = -0.25;

//Control para el motor 6
int M6Dir = 0; //0->STOP 1->FWRD 2-> BWRD
int M6Pos = 0;
float M6PosV = 0;
float M6PosR = 0;
float M6Error = 0;
float M6PastError = 0;
float M6DerError = 0;
float M6PastDerError = 0;
float M6W = 0.5;
float M6Control = 0;
float M6ControlValue = 0;
float M6Ref = 0;

float M6a = 0.0628;
float M6b = 1.7099;
float M6c = -0.12;

void setup() {
  Serial.begin(250000); // Set the baud rate to match the Python application

  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M1PWM, OUTPUT);
  pinMode(M1Pot, INPUT);

  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);
  pinMode(M2PWM, OUTPUT);
  pinMode(M2Pot, INPUT);

  pinMode(M3A, OUTPUT);
  pinMode(M3B, OUTPUT);
  pinMode(M3PWM, OUTPUT);
  pinMode(M3Pot, INPUT);

  pinMode(M4A, OUTPUT);
  pinMode(M4B, OUTPUT);
  pinMode(M4PWM, OUTPUT);
  pinMode(M4Pot, INPUT);

  pinMode(M5A, OUTPUT);
  pinMode(M5B, OUTPUT);
  pinMode(M5PWM, OUTPUT);
  pinMode(M5Pot, INPUT);

  pinMode(M6A, OUTPUT);
  pinMode(M6B, OUTPUT);
  pinMode(M6PWM, OUTPUT);
  pinMode(M6Pot, INPUT);
  
}

void loop() {

  if (Serial.available()) {
    String receivedData = Serial.readStringUntil('\n'); // Read the entire line until a newline character
    int valueIndex = 0;
    char *token = strtok((char *)receivedData.c_str(), ",");
    
    while (token != NULL && valueIndex < 6) {
      receivedValues[valueIndex] = atoi(token);
      valueIndex++;
      token = strtok(NULL, ",");
    }

    if (valueIndex == 6) {
      Serial.print("A");

      M6Control = receivedValues[0]-255;
      
      if (M6Control > 0)
      {
        digitalWrite(M6A,HIGH);
        digitalWrite(M6B,LOW);
      }
      else if(M6Control < 0)
      {
        digitalWrite(M6A,LOW);
        digitalWrite(M6B,HIGH);
      }
      else
      {
        digitalWrite(M6A,LOW);
        digitalWrite(M6B,LOW);
      }
      
      

      
      analogWrite(M6PWM,abs(M6Control));
      
      M6Pos = analogRead(M6Pot);
      M6PosV = (M6Pos * 5.0) / 1023;
      M6PosR = (M6a*M6PosV*M6PosV) + M6b*M6PosV + M6c;
      Serial.print(M6PosR);

      Serial.print(", ");
      Serial.print(0);
      analogWrite(M2PWM,0);
      Serial.print(", ");
      Serial.print(0);
      analogWrite(M3PWM,0);
      Serial.print(", ");
      Serial.print(0);
      analogWrite(M4PWM,0);
      Serial.print(", ");
      Serial.print(0);
      analogWrite(M5PWM,0);
      Serial.print(", ");
      Serial.println(0);
      analogWrite(M1PWM,0);
    }
  }
}
