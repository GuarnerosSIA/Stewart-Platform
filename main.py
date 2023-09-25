import serial
import time
import numpy as np

# Initialize the serial connection
ser = serial.Serial('COM3', 250000)  # Replace 'COM1' with your Arduino's serial port
time.sleep(3)

frecuencia = 0.5
dt = 0.005
tiempo = np.linspace(0,5,int(5/dt), endpoint= False)

tic = time.time()
for i in tiempo:
# Send integers to Arduino
    integers_to_send = [int(1*(50*np.sin(2*np.pi*i*frecuencia)+255)),
                         0, 
                         0,
                         0,
                         0,
                         0]
    data_to_send = ','.join(map(str, integers_to_send)) + '\n'
    ser.write(data_to_send.encode('utf-8'))
    A = ser.readline()
    actuators = A.decode('utf-8')
    if actuators[0]=='A':
        act_sep = actuators[1:].replace('\r\n','').split(',')
        measurements = np.array([float(item) for item in act_sep])
        
        #print(measurements)
    #time.sleep(dt)

# Close the serial connection
toc = time.time() - tic
print(toc)
ser.close()