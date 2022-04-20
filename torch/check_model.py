iimport torch
import argparse

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    # 30M float params roughly equals to 120M model size
    assert total < 30 * 1e6, 'total params : ' + str(total / 1e6) + 'M'

def count_flops_and_shape(model, track):
    flops_dict = {}
    def make_conv2d_hook(name):
        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[0] / m.stride[1] / m.groups
            flops_dict[name] = int(flops)
        return conv2d_hook

    def make_fc_hook(name):
        def fc_hook(m, input):
            prod = 1
            for dim in input[0].size()[1:]:  # exclude batch size
                prod *= dim
            flops = prod * m.out_features
            flops_dict[name] = int(flops)
        return fc_hook

    def make_bn_hook(name):
        def bn_hook(m, input):
            prod = 1
            for dim in input[0].size()[1:]:  # exclude batch size
                prod *= dim
            flops = prod * 2
            flops_dict[name] = int(flops)
        return bn_hook
            
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, torch.nn.Linear):
            h = m.register_forward_pre_hook(make_fc_hook(name))
            hooks.append(h)
        elif isinstance(m, torch.nn.BatchNorm2d):
            h = m.register_forward_pre_hook(make_bn_hook(name))
            hooks.append(h)

    input = torch.zeros(1,3,224,224)
    model.eval()
    with torch.no_grad():
        output = model(input)
    if track == 1:
        assert output.numel() == 100
    else:
        assert output.numel() == 1

    total_flops = 0
    for k, v in flops_dict.items():
        total_flops += v
    assert total_flops < 5 * 1e9, 'Flops: ' + str(total_flops / 1e9) + 'G'


def main():
    parser = argparse.ArgumentParser(description='Adversarial Solver')
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--track', required=True, type=int)
    args = parser.parse_args()
    assert args.track in [1, 2]
    import sys
    sys.path.append(args.model) 
    from model import Model
    model = Model().float()
    count_params(model)
    count_flops_and_shape(model, args.track)

if __name__ == '__main__':
    main()

