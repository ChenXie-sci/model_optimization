# tensorflow ----> h5
# pytorch    ----> pth
#
# middle layer: ONNX

import torch

x = torch.randn(1, 3, 256, 256) # image tensor
model = model() 
 
with torch.no_grad(): 
    torch.onnx.export( 
        model, 
        x, 
        "srcnn.onnx", 
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'])


# ONNX checker
import onnx 
 
onnx_model = onnx.load("srcnn.onnx") 
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")


# model deployment
import onnxruntime 
import numpy as np
import opencv as cv2 

ort_session = onnxruntime.InferenceSession("srcnn.onnx") 
ort_inputs = {'input': input_img} 
ort_output = ort_session.run(['output'], ort_inputs)[0] 
 
ort_output = np.squeeze(ort_output, 0) 
ort_output = np.clip(ort_output, 0, 255) 
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
cv2.imwrite("image.png", ort_output)





