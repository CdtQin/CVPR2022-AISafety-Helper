import onnx
import onnxruntime
import numpy as np
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Adversarial Solver')
    parser.add_argument('--torch_model', required=True, type=str)
    parser.add_argument('--onnx_model', required=True, type=str)

    args = parser.parse_args()

    im = np.random.randn(*[1, 3, 224, 224]).astype(np.float32)
    onnx_model = onnx.load(args.onnx_model)
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    ort_inputs = {}
    for i, input_ele in enumerate(ort_session.get_inputs()):
        ort_inputs[input_ele.name] = im
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)
    print('onnx_output:', ort_outs)

    import sys
    sys.path.append(args.torch_model) 
    from model import Model
    torch_model = Model().eval()
    torch_outs = torch_model(torch.from_numpy(im))
    print('torch_output:', torch_outs)

if __name__ == '__main__':
    main()
