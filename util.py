import torch


def is_close(prev_param, new_param, eps=1e-6):
    delta = torch.max(torch.abs(new_param - prev_param)).item()
    print(delta)

    return delta < eps


def input_to_tensor(inp, dtype=torch.float):
    if torch.is_tensor(inp):
        inp = inp.type(dtype)
    else:
        inp = torch.from_numpy(inp).type(dtype)

    return inp
