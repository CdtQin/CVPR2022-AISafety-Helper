import onnx
import onnx.numpy_helper
import torch

from op_code_generators import OpCodeGenerator


class InstanceNormalizationOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(InstanceNormalizationOpCodeGenerator,
          self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2

    view = False
    if d == 0:
      d = 1
      view = True

    nn_name = f"InstanceNorm{d}d"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)

    params_str = self.gen_params_str(num_features=onnx.numpy_helper.to_array(
        initializers[node.input[1]]).shape[0],
                                     eps=attr_value_dict["epsilon"])

    init_str, forward_str = [], []
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(f"self.{node_name}.weight = {inputs_str[1]}")
    init_str.append(f"self.{node_name}.bias = {inputs_str[2]}")
    curr_input = inputs_str[0]
    if view:
      forward_str.append(f"{curr_input} = torch.unsqueeze({curr_input}, -1)")
    forward_str.append(f"{outputs_str[0]} = self.{node_name}({curr_input})")
    if view:
      forward_str.append(
          f"{outputs_str[0]} = torch.squeeze({outputs_str[0]}, -1)")
    return {"init": init_str, "forward": forward_str}
