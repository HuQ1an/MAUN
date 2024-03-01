import torch
from .MAUN import MAUN
def model_generator(method, pretrained_model_path=None):
    if 'MAUN' in method:
        num_iterations = int(method.split('_')[1][:-3])
        print(num_iterations)
        model = MAUN(num_iterations=num_iterations).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model