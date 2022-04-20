import torch.onnx

x = torch.randn(1, 3, 224, 224, requires_grad=True)

def submit_model():
    model = model_construct()
    return model

def load_state(path, model):
    pass

torch_model = submit_model()
load_path = 'ckpt.pth'
load_state(load_path, torch_model)

load_state(load_path, torch_model)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "submit.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])
