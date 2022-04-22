import torch
import argparse
from ptflops.flops_counter import get_model_complexity_info

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
    flops, params = get_model_complexity_info(m, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    assert params < 30 * 1e6, 'total params : ' + str(params / 1e6) + 'M'
    assert flops < 5 * 1e9, 'Flops: ' + str(flops / 1e9) + 'G'
    input = torch.zeros(1,3,224,224)
    model.eval()
    with torch.no_grad():
        output = model(input)
    if track == 1:
        assert output.numel() == 100
    else:
        assert output.numel() == 1
    

if __name__ == '__main__':
    main()

