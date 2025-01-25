##### Load and test with onnx
import onnx
import onnxruntime as ort
import numpy as np


# Variables comunes
actor_State = 'input_1'
critic_State = 'input_1'
critic_Action = 'input_2'

steps = 10

# Cargo las redes
path_actor = "C:\\Users\\guarn\\Dropbox\\Alejandro\\DoctoradoITESM\\Overleaf\\SGP control\\Stewart platform physical control\\actorRLSGP.onnx"
model_actor = onnx.load(path_actor)
onnx.checker.check_model(model_actor)
actor_nn = ort.InferenceSession(path_actor)

path_critic = "C:\\Users\\guarn\\Dropbox\\Alejandro\\DoctoradoITESM\\Overleaf\\SGP control\\Stewart platform physical control\\criticRLSGP1.onnx"
model_critic = onnx.load(path_critic)
onnx.checker.check_model(model_critic)
critic_nn = ort.InferenceSession(path_critic)

#####
# SimulacionPlataforma


# calculo reward
# valores de los pesos

# outputActor = actor_nn.run(None,{actor_State:observation})
# outputCritic = critic_nn.run(None,{critic_State:observation,
#                                    critic_Action:states})



#States-Actions