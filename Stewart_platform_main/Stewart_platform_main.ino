int receivedValues[6]; // Array to store the received integer values

#define MIN_CONTROL_VALUE 25

#define M1A 3
#define M1B 2
#define M1Pot A0

#define M2A 5
#define M2B 4
#define M2Pot A1

#define M3A 7
#define M3B 6
#define M3Pot A2

#define M4A 9
#define M4B 8
#define M4Pot A3

#define M5A 11
#define M5B 10
#define M5Pot A4

#define M6A 45
#define M6B 44
#define M6Pot A5


//motor 1
int M1Pos = 0;
float M1PosV = 0;
float M1PosR = 0;
float M1Control = 0;

float M1a = 0.0876;
float M1b = 1.6039;
float M1c = -0.21;

//motor 2
int M2Pos = 0;
float M2PosV = 0;
float M2PosR = 0;
float M2Control = 0;

float M2a = 0.06282;
float M2b = 1.7098;
float M2c = -0.12;

//motor 3
int M3Pos = 0;
float M3PosV = 0;
float M3PosR = 0;
float M3Control = 0;

float M3a = 0.06866;
float M3b = 1.70467;
float M3c = -0.24;

//motor 4
int M4Pos = 0;
float M4PosV = 0;
float M4PosR = 0;
float M4Control = 0;

float M4a = 0.0016;
float M4b = 1.9999;
float M4c = -0.14;

//motor 5
int M5Pos = 0;
float M5PosV = 0;
float M5PosR = 0;
float M5Control = 0;

float M5a = 0.07903;
float M5b = 1.6548;
float M5c = -0.25;

//motor 6
int M6Pos = 0;
float M6PosV = 0;
float M6PosR = 0;
float M6Control = 0;

float M6a = 0.0628;
float M6b = 1.7099;
float M6c = -0.12;

// Counting

int cont_com = 0;
int threshold_cont = 1000;

void setup() {
  Serial.begin(250000); // Set the baud rate to match the Python application

  

  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M1Pot, INPUT);

  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);
  pinMode(M2Pot, INPUT);

  pinMode(M3A, OUTPUT);
  pinMode(M3B, OUTPUT);
  pinMode(M3Pot, INPUT);

  pinMode(M4A, OUTPUT);
  pinMode(M4B, OUTPUT);
  pinMode(M4Pot, INPUT);

  pinMode(M5A, OUTPUT);
  pinMode(M5B, OUTPUT);
  pinMode(M5Pot, INPUT);

  pinMode(M6A, OUTPUT);
  pinMode(M6B, OUTPUT);
  pinMode(M6Pot, INPUT);

  
}

void loop() {
  cont_com += 1;
  if (cont_com >= threshold_cont)
  {
    cont_com = threshold_cont;
    analogWrite(M1A,0);
    analogWrite(M1B,0);

    analogWrite(M2A,0);
    analogWrite(M2B,0);

    analogWrite(M3A,0);
    analogWrite(M3B,0);

    analogWrite(M4A,0);
    analogWrite(M4B,0);

    analogWrite(M5A,0);
    analogWrite(M5B,0);

    analogWrite(M6A,0);
    analogWrite(M6B,0);
  }

  if (Serial.available()) {
    cont_com = 0;
    String receivedData = Serial.readStringUntil('\n'); // Read the entire line until a newline character
    int valueIndex = 0;
    char *token = strtok((char *)receivedData.c_str(), ",");
    
    while (token != NULL && valueIndex < 6) {
      receivedValues[valueIndex] = atoi(token);
      valueIndex++;
      token = strtok(NULL, ",");
    }

    if (valueIndex == 6) {


      M1Control = receivedValues[0]-255;
      M2Control = receivedValues[1]-255;

      M3Control = receivedValues[2]-255;
      M4Control = receivedValues[3]-255;

      M5Control = receivedValues[4]-255;
      M6Control = receivedValues[5]-255;


      pwmWrite(M1Control, M1A, M1B, MIN_CONTROL_VALUE);
      pwmWrite(M2Control, M2A, M2B, MIN_CONTROL_VALUE);

      pwmWrite(M3Control, M3A, M3B, MIN_CONTROL_VALUE);
      pwmWrite(M4Control, M4A, M4B, MIN_CONTROL_VALUE);

      pwmWrite(M5Control, M5A, M5B, MIN_CONTROL_VALUE);
      pwmWrite(M6Control, M6A, M6B, MIN_CONTROL_VALUE);

      M1Pos = readPot(M1Pot);
      M1PosV = (M1Pos * 5.0) / 1023;
      M1PosR = (M1a*M1PosV*M1PosV) + M1b*M1PosV + M1c;
      M2Pos = readPot(M2Pot);
      M2PosV = (M2Pos * 5.0) / 1023;
      M2PosR = (M2a*M2PosV*M2PosV) + M2b*M2PosV + M2c;

      M3Pos = readPot(M3Pot);
      M3PosV = (M3Pos * 5.0) / 1023;
      M3PosR = (M3a*M3PosV*M3PosV) + M3b*M3PosV + M3c;
      M4Pos = readPot(M4Pot);
      M4PosV = (M4Pos * 5.0) / 1023;
      M4PosR = (M4a*M4PosV*M4PosV) + M4b*M4PosV + M4c;
      
      M5Pos = readPot(M5Pot);
      M5PosV = (M5Pos * 5.0) / 1023;
      M5PosR = (M5a*M5PosV*M5PosV) + M5b*M5PosV + M5c;
      M6Pos = readPot(M6Pot);
      M6PosV = (M6Pos * 5.0) / 1023;
      M6PosR = (M6a*M6PosV*M6PosV) + M6b*M6PosV + M6c;
      
      Serial.print("A");
      Serial.print(M1PosR);
      Serial.print(", ");
      Serial.print(M2PosR);
      Serial.print(", ");
      Serial.print(M3PosR);
      Serial.print(", ");
      Serial.print(M4PosR);
      Serial.print(", ");
      Serial.print(M5PosR);
      Serial.print(", ");
      Serial.println(M6PosR);
    }
  }
}


float readPot(int pot_number)
{
  /*
  This function reads analog ports and compute the average of the readings
  */
  int readings = 5;
  float sum = 0;
  for(int i = 0; i <= readings; i++)
  {
    sum += analogRead(pot_number);
  }
  return sum/readings;
}

void pwmWrite(int control, int motorA, int motorB, int threshold)
{
  /*
  This function receives the control computed and send PWM to the corresponding H-bridge  channels.
  It takes into account a threshold, when the control is less than it, both channels are set to zero.
  Otherwise, a PWM is sento to the computed direction.
  */
  if (abs(control) < threshold)
  {
    control = 0;
  }
  if (control > 0)
  {
    analogWrite(motorA,0);
    analogWrite(motorB,abs(control));
  }
  else if(control < 0)
  {
    analogWrite(motorA,abs(control));
    analogWrite(motorB,0);
  }
  else
  {
    analogWrite(motorA,0);
    analogWrite(motorB,0);
  }
}

float motorMeasurement(float motorPosition, float motorCoeffA, float motorCoeffB, float motorCoeffC)
{
  float motorVoltage = (motorPosition * 5.0) / 1023;
  return (motorCoeffA*motorVoltage*motorVoltage) + motorCoeffB*motorVoltage + motorCoeffC;
}
