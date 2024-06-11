from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

leftHandle = sim.getObject('./Revolute_joint[0]/leftDown/Prismatic_joint')
# leftHandle = sim.getObject('./dyorBaseDyn/leftMotor')
# rightHandle = sim.getObject('./dyorBaseDyn/rightMotor')

sim.startSimulation()
while (t := sim.getSimulationTime()) < 100:
    print(f'Simulation time: {t:.2f} [s]')
    sim.setJointTargetForce(leftHandle,math.sin(t)*31)
    # sim.setJointTargetVelocity(leftHandle,0)
    sim.step()
sim.stopSimulation()