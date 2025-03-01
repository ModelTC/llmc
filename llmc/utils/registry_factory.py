class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target):
        return self.register(target)

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f'Error:{value} must be callable!')
            if key in self._dict:
                raise Exception(f'{key} already exists.')
            self[key] = value
            return value

        if callable(target):
            return add_item(target.__name__, target)
        else:
            return lambda x: add_item(target, x)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


PREPROC_REGISTRY = Register()
ALGO_REGISTRY = Register()
MODEL_REGISTRY = Register()
KV_REGISTRY = Register()
TOKEN_REDUCTION_REGISTRY = Register()
