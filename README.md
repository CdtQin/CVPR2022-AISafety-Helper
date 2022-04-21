# CVPR2022-AISafety-Helper

### Submission Instructions

Your submission should be either an onnx file or a zip file of pytorch model.
  
For the first option, onnx/to_onnx.py is the tool to convert your pytorch model to onnx model. And we will use onnx/onnx2torch.sh to convert it back. This tool is modified from https://github.com/ToriML/onnx2pytorch. The strength of this option is a clean interface, while the weakness is the imperfect conversion process. For some complex model, the conversion may not be correct. To check if pytorch model aligns with onnx model, see the inference example: onnx/inference.py
  
For the second option, we require a zip file, which directly contains a file named model.py. model.py should include class Model(). In other words, the zip file architecture should be:
	
```Shell
.
├── model.py
├── ...
└── any other files
```

an simple example of model.py can be found at submit/model.py . We decompress your file with following code:
```Shell
shutil.unpack_archive(your_zip_file_name, model_torch_dir, "zip")
```

Based on the rule of model size, the submitted model should contain no more than 30M Float parameters (120Mb in model size), and its operations should be lower than 5G FLOPs. We release the check script at torch/check_model.py (currently, the code is not robust, and we are going to use ptflops, we will update the file while we finished test on our server.). In this file, you can see how the model is loaded:

```Shell
import sys
sys.path.append(args.model) 
from model import Model
model = Model().float()
```

### Model Inference

The input of your model is a tensor whose shape is Nx3x224x224. The range of the tensor is 0 to 1, and the channel order is BGR. If you prefer your own normalization hyper-parameter, you can conduct the conversion in your own model. For Track1, your output should be Nx100, while for Track2, the output should be Nx1.

### No a NN Methods

For those participants who do not utilize NN method, mainly for Track2, your method can be packed into the forward function in submit/model.py .
Note that you should maintain some paramters in the model, which is required by DistributedDataParallel of PyTorch. Otherwise, your submission will encounter:

```
assertionerror: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient
```

### Track2 Evaluation

For track2, your model should predict a confidence score for each input image. We will first sort the scores of the whole dataset, and then find out the best threshold that yields the highest F1-Score. And this F1-Score will be your final score.


### Environment

We list our docker enviroment in env.txt . You should check the enviroment if you choose the zip way to submit your results.
