def append_method(cls, func):
    def method(self, *args, **kwargs):
        return func(*args, **kwargs)

    method.__name__ = func.__name__
    setattr(cls, func.__name__, method)
