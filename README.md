# CVPR2022-AISafety-Helper

### Submission Instructions

Your submission should be either an onnx file or a zip file of pytorch model.
  
For the first option, onnx/to_onnx.py is the tool to convert your pytorch model to onnx model. And we will use onnx/onnx2torch.sh to convert it back. This tool is modified from https://github.com/ToriML/onnx2pytorch. The strength of this option is a clean interface, while the weakness is the imperfect conversion process. For some complex model, the conversion may not be correct.
  
For the second option, we require the zip file contain a folder named submit. It should contain a file named model.py, which includes class Model(). In other words, the folder architecture should be:
	
```Shell
.
├── submit
    ├── model.py
    ├── ...
    └── any other files
```

an simple example of model.py can be found at submit/model.py

Based on the rule of model size, the submitted model should contain no more than 30M Float parameters (120Mb in model size), and its operations should be lower than 5G FLOPs. We release the check script at torch/check_model.py . In this file, you can see how the model is loaded:

```Shell
import sys
sys.path.append(args.model) 
from model import Model
model = Model().float()
```

### Environment

We list our docker enviroment in env.txt . You should check the enviroment if you choose the zip way to submit your results.
