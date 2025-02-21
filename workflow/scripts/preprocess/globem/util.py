from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def progress_decorator(func):
    def wrapper(self, *args, **kwargs):
        action_name = func.__name__.replace("_", " ").capitalize()
        print(f"{'='*20} Starting: {action_name} {'='*10}")
        result = func(self, *args, **kwargs)
        print(f"{'='*20} Completed: {action_name} {'='*10}")
        return result

    return wrapper
