import onnx
import torch

from op_code_generators import OpCodeGenerator


class ExpandOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ExpandOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    forward_str.append(f"{outputs_str[0]} = {inputs_str[0]}.expand(*[i if i != 1 else -1 for i in list({inputs_str[1]})])")
    return {"init": init_str, "forward": forward_str}
