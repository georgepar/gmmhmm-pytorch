import torch


def input_to_tensor(inp, dtype=torch.float):
    if torch.is_tensor(inp):
        inp = inp.type(dtype)
    else:
        inp = torch.from_numpy(inp).type(dtype)

    return inp
