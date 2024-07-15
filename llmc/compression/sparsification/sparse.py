class Sparser:
    def __init__(self, sparsity, **kwargs):
        self.sparsity = sparsity
        self.kwargs = kwargs
        self.W_mask = None
