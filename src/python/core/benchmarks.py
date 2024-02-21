import logging
from functools import wraps
import time


def timeit(name: str):
    def decorator(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logging.info(f'Function {name} Took {total_time:.4f} seconds')
            return result
        return timeit_wrapper

    return decorator
