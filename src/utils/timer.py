import time


class Timer:
    

    def __init__(self, unit: str = 's', name: str = None):
        
        self.name = name
        self.unit = unit

    def __call__(self, func):
        
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000 if self.unit == 'ms' else (time.perf_counter() - start)
            name = self.name if self.name else func.__name__
            
            return result

        return wrapper
