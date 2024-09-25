# !pip install pydantic==4.5.0
# !pip install onnx
# !pip install onnxruntime
# !pip install pip install git+https://github.com/onnx/onnx-tensorflow.git
# !pip install tf-nightly

import tensorflow as tf
#import onnx
import cv2
import os
#import onnxruntime
import torch
import torch.nn as nn
import numpy as np


# for quantization needs
import torch.quantization as quantization
import torch.nn.quantized as nnq
from torch.quantization import get_default_qconfig, prepare, convert
torch.backends.quantized.engine = 'qnnpack'



#%cd deep-text-recognition-benchmark
from model import Model

#from onnx_tf.backend import prepare
#from pydantic import BaseModel
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'



data = {
        'image_folder': 'demo_image/', 
        'saved_model' : '../models/lcd_ocr_vgg_bilstm_trial/best_accuracy.pth'
        }


# class InputData(BaseModel):
#     image_folder: str
#     saved_model: str
#     batch_max_length: int = 9 #6
#     imgH: int = 100 #120
#     imgW: int = 400 #360
#     character: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
#     Transformation: str = 'None'
#     FeatureExtraction: str = 'VGG'
#     SequenceModeling: str = 'None'
#     Prediction:str = 'CTC'
#     num_fiducial: int = 20
#     input_channel: int = 1
#     output_channel: int = 512
#     hidden_size: int = 256
#     num_class = 38
#     batch_size: int = 1
    
# opt = InputData(**data)

#model = Model(opt)


#text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)

# Representative Data Generation Function used for Integer Quantization
dataset_path = 'represent_data/'
def representative_data_gen():
    for file in os.listdir(dataset_path):
        image_path = dataset_path + file
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        image = image/127.5 - 1.0
        image = cv2.resize(image,(400, 100),interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = np.float32(image)
        yield [image]



def load_torch_model(torch_model_path, as_model = False):

    #model = Model(opt)
    # if as_model:
    #     model = torch.load(torch_model_path, map_location = device)
        
    # else:
    def fix_model_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]  
            new_state_dict[key] = value
        return new_state_dict

    # # load model
    # print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(fix_model_state_dict(torch.load(torch_model_path, map_location=device)))
    model.eval()
        
    return model



def export_onnx_model(destpath, imgh = 120, imgw = 360):
    input = torch.randn(1, 1, imgh, imgw).detach()
    #if device == 'cuda':
    #    input = input.half()


    with torch.no_grad():
        # Export the model
        torch.onnx.export(model,               # model being run
                        (input,text_for_pred),                         # model input (or a tuple for multiple inputs)
                        destpath,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        
        #              dynamic_axes={'input' : {0:'batch_size',}}
        #                                       1:'channel',
        #                                      2 : 'height',
        #                                       3 : 'width'},    # variable length axes
        #             #                         'output' : {1 : 'seq_length'}}
        #              }
         )
    print("Model converted from torch to onnx succesfully\n=============================================\n")
    
    
def load_input_image(image_path ='../YRBybNcARj_1717501021400.jpg', imgh=120, imgw = 360, dtype ='float32'):
    '''
    Returns np array image
    '''
    #GT for uxAyC9Ylhx_1700481287972.jpg: BT23VMF
    #image_path =    #'/content/deep-text-recognition-benchmark/demo_image/demo_4.png'
    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image = image/127.5 - 1.0
    image = cv2.resize(image,(imgw, imgh),interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    
    if dtype == "float16":
        image = np.float16(image)
    elif dtype == "float32":
        image = np.float32(image)
    elif dtype == 'int8':
        image = np.int8(image)
    
        
    return image

    
def onnx_inference(model, onnx_model_path, imgpath, imgh, imgw):
    '''
    model: torch model
    '''
    image = load_input_image(image_path=imgpath, imgh = imgh, imgw = imgw)
    print(f"input image shape: {image.shape}")

    torch_input = torch.from_numpy(image)
    
    print(f"torch input shape: {torch_input.shape}")
    
    print(f"onnx_model_path: {onnx_model_path}")
    
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    print(f"ort_session.get_inputs(): {ort_session.get_inputs()}")

    # # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Run PyTorch model
    store_out = model(torch_input, text_for_pred)
    
    # Check if torch and onnx output are equal
    np.testing.assert_allclose(store_out.detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-04)
    print("torch and onnx output Comparison Succesful\n=============================================\n")
    
    
def convert_onnx_to_pb(onnx_model_path, dest_path):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    print("Exporting to pb model")
    tf_rep.export_graph(dest_path)
    print(f"Tensorflow model exported to {dest_path}\n=============================================\n")
    
    
# Download and unzipping representative dataset#%%bash
#!wget https://github.com/tulasiram58827/ocr_tflite/raw/main/data/represent_data.zip
#!unzip represent_data.zip
def convert_to_tflite(pb_model_path, destpath, tf_input_shape = (1,1,120,360), quantization='dr'):

    loaded = tf.saved_model.load(pb_model_path)
    concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    concrete_func.inputs[0].set_shape(tf_input_shape)
    
    if quantization == 'float16':
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization == 'int8' or quantization == 'full_int8':
        converter.representative_dataset = representative_data_gen
        
    if quantization == 'full_int8':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, 
                                               ]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        
    tflite_model = converter.convert()
    model_name = destpath #f'crnn_{quantization}.tflite'
    with open(model_name, 'wb') as f:
      f.write(tflite_model)
      
    print(f"tflite model successfully saved to {destpath}\n=============================================\n")
    
    
def tflite_inference(model_name, input, quantization='dr'):
    #model_name = f'crnn_{quantization}.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_name)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def ctc_decode(preds):
    print(f"Input to ctc_decode: {preds.shape}")
    pred_index = np.argmax(preds, axis=2)
    #char_list = list(opt.character)
    char_list = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
    char_dict = {}
    for i, char in enumerate(char_list):
        char_dict[char] = i + 1
    char_list = ['_'] + char_list
    BLANK = 0
    texts = []
    output = pred_index[0, :]
    characters = []
    for i in range(preds.shape[1]):
        if output[i] != BLANK and (not (i > 0 and output[i - 1] == output[i])):
            characters.append(char_list[output[i]])
        text = ''.join(characters)
    return text


if __name__ == '__main__':
    image_height = 100
    image_width = 400
    tflite_quantization = 'full_int8'
    imgpath = "../0L3TskzPbM_1699616354236.jpg"
    
    
    perform_torch_to_onnx_conversion = False
    perform_onnx_inference = False
    perform_onnx_to_pb_conversion = False
    perform_tf_to_tflite_conversion = False
    perform_tflite_inference = True
    
    
    #Model paths
    #torch_model_path = '../models/lcd_ocr_vgg_bilstm_trial/best_accuracy.pth'
    torch_model_path = "results/lp_ocr_v1/best_accuracy.pth"
    onnx_destpath = 'results/lp_ocr_v1/lpocrv1.onnx'
    tf_destpath = 'results/lp_ocr_v1/lpocrv1.pb'
    tflite_destpath = 'results/lp_ocr_v1/lpocrv1.tflite'
    
    
    #Load torch model    
    #torch_model = load_torch_model(torch_model_path, as_model = False)
    
    
    # Convert torch to onnx
    if perform_torch_to_onnx_conversion:
        export_onnx_model(onnx_destpath, imgh = image_height, imgw = image_width)
    
        
    # Run inference with torch and onnx model and compare outputs
    if perform_onnx_inference:
        onnx_inference(torch_model, onnx_destpath, imgpath, image_height, image_width)
    
    
    # Convert onnx to pb(tensorflow) model
    if perform_onnx_to_pb_conversion:
        convert_onnx_to_pb(onnx_destpath, tf_destpath)
        
    
    # Convert pb (tensorflow) to tflite model
    if perform_tf_to_tflite_conversion:
        convert_to_tflite(tf_destpath, tflite_destpath,tf_input_shape = (1,1,image_height,image_width),
                          quantization=tflite_quantization)
    
    
    # tflite inference
    if perform_tflite_inference:
        image = load_input_image(image_path = "../0L3TskzPbM_1699616354236.jpg", imgh=image_height, imgw = image_width, dtype='int8')
        preds = tflite_inference(tflite_destpath, image)
        final_text = ctc_decode(preds)
        
        print(f"Final Prediction: {final_text}")