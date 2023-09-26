import serial
import time
import numpy as np

# Initialize the serial connection
ser = serial.Serial('COM3', 250000)  # Replace 'COM1' with your Arduino's serial port
time.sleep(3)

def control_bounds(x):
    if x >= 255:
        x = 255
    elif x <= -255:
        x = -255
    return x + 255
vControlBound = np.vectorize(control_bounds)


positions = np.ones((1,6))*5
print(positions)

frecuencia = 0.5
dt = 0.005
tiempo = np.linspace(0,5,int(5/dt), endpoint= False)

kp = np.array([
    [0,0,0,0,0,50]
    ])

control = np.zeros((1,6))

tic = time.time()

for i in tiempo:
# Send integers to Arduino
    integers_to_send = [int(control[0,0]),
                         0, 
                         0,
                         0,
                         0,
                         int(control[0,5])]
    data_to_send = ','.join(map(str, integers_to_send)) + '\n'
    ser.write(data_to_send.encode('utf-8'))
    A = ser.readline()
    actuators = A.decode('utf-8')
    if actuators[0]=='A':
        act_sep = actuators[1:].replace('\r\n','').split(',')
        measurements = np.array([float(item) for item in act_sep])
        deltas = positions - measurements
        control = np.multiply(deltas,kp)
        control = vControlBound(control)
        print(measurements)
    #time.sleep(dt)


# Close the serial connection
toc = time.time() - tic
print(toc)
ser.close()