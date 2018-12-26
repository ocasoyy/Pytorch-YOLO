from yolo_config import *

def space_to_depth(x, block_size=2):
    out = x.permute(0, 2, 3, 1)
    batch_size, s_height, s_width, s_depth = out.size()
    d_depth = s_depth * (block_size ** 2)
    d_height = s_height // block_size
    t_1 = out.split(block_size, 2)
    stacked = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
    out = torch.stack(stacked, 1)
    out = out.permute(0, 2, 1, 3)
    out = out.permute(0, 3, 1, 2)
    return out

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
