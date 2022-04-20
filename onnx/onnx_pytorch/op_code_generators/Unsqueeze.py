import onnx
import torch
from onnx.numpy_helper import to_array

from op_code_generators import OpCodeGenerator


class UnsqueezeOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(UnsqueezeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    axes = attr_value_dict.get("axes", [])
    if len(node.input) == 2:
      assert node.input[
          1] in initializers, "Currently UnsqueezeOpCodeGenerator only support all of [axes] is in initializers."
      axes = to_array(initializers[node.input[1]])
    init_str, forward_str = [], []
    curr_input = inputs_str[0]
    for a in axes:
      forward_str.append(
          f"{outputs_str[0]} = torch.unsqueeze({curr_input}, {a})")
      curr_input = outputs_str[0]

    return {"init": init_str, "forward": forward_str}
