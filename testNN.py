##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np
import time

onnx_path_actor = "C:\\Users\\guarn\\Dropbox\\Alejandro\\DoctoradoITESM\\Overleaf\\SGP control\\Stewart platform physical control\\actorRLSGP.onnx"
onnx_model_actor = onnx.load(onnx_path_actor)
onnx.checker.check_model(onnx_model_actor)
ort_sess_actor = ort.InferenceSession(onnx_path_actor)
input_name_actor = ort_sess_actor.get_inputs()[0].name
input_shape_actor = ort_sess_actor.get_inputs()[0].shape

input_data_actor = np.random.random((1,10)).astype(np.float32)
output = ort_sess_actor.run(None,{input_name_actor:input_data_actor})

print("Model Output: ", output)
################################################################

onnx_path_critic = "C:\\Users\\guarn\\Dropbox\\Alejandro\\DoctoradoITESM\\Overleaf\\SGP control\\Stewart platform physical control\\criticRLSGP1.onnx"
onnx_model_critic = onnx.load(onnx_path_critic)
onnx.checker.check_model(onnx_model_critic)
ort_sess_critic = ort.InferenceSession(onnx_path_critic)
input_shape_critic = ort_sess_critic.get_inputs()[0].shape
input_shape_critic = ort_sess_critic.get_inputs()[1].shape

input_state_critic = np.random.random((1,10)).astype(np.float32)
input_action_critic = np.random.random((1,10)).astype(np.float32)

output = ort_sess_critic.run(None,{'input_1':input_state_critic,
                                   'input_2':input_action_critic})

print("Model Output: ", output)


#States-Actions