"""
decorators
"""
from tensorboardX import SummaryWriter
from tqdm import tqdm

# decorators for tqdm
# Todo


# decorators for tensorboard
def tensorboard_decorator(tblogpath:str=None):
    """tensorboard decorator
    Args:
        tblogpath (str): tensorboard log path
    Notice:
        the function was decorated must output a int value count the game episode
    """
    tbwriter = SummaryWriter(tblogpath)
    def decorator(f):
        def warpped_function(*args, **kwargs):
            got = f(*args, **kwargs)
            for item in got:
                tbwriter.add_scalar(item, got[item], args[0])
        return warpped_function
    tbwriter.close()
    return decorator