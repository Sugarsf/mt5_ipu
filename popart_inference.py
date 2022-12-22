import popart
import torch.onnx
import numpy as np
import onnx


loaded_model = onnx.load('mt5.onnx')

inputs_name = [node.name for node in loaded_model.graph.input]
outputs_name = [node.name for node in loaded_model.graph.output]

print("Iputs name:", inputs_name)   
print("Outputs name:", outputs_name)



anchors = {outputs_name[0]:popart.AnchorReturnType("All")}
           
dataFlow = popart.DataFlow(1, anchors)
device = popart.DeviceManager().createCpuDevice()

session = popart.InferenceSession("mt5.onnx", dataFlow, device)
session.prepareDevice()

anchors1 = session.initAnchorArrays()

input_1 = np.random.random((1,9)).astype(np.int32)
input_2 = np.random.random((1,9)).astype(np.int32)
input_3 = np.random.random((1,6)).astype(np.int32)

stepio = popart.PyStepIO({inputs_name[0]: input_1,inputs_name[1]:input_2,inputs_name[2]:input_3}, anchors1)

session.run(stepio)

print (anchors1)
for key in anchors1:
 res = anchors1[key]
 print (res.shape)
 print (res[0][0:30])
 exit()