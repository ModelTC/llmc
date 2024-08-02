class Sparser:
    def __init__(self, sparsity_constraint, **kwargs):
        if 'sparsity' in sparsity_constraint:
            self.sparsity = sparsity_constraint['sparsity']
            self.W_mask = None
        elif 'n_prune_layers' in sparsity_constraint:
            self.n_prune_layers = sparsity_constraint['n_prune_layers']
            self.importances = None
        self.kwargs = kwargs
