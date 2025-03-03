import torch.nn as nn

class PairNorm(nn.Module):
    def __init__(self, mode='PN-SCS', scale=1.0):
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale
        
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class DynamicPairNorm(nn.Module):
    def __init__(self):
        super(DynamicPairNorm, self).__init__()
    
    def __TransFeauture(self):
        """
        translate the feature of node to pair distance, so that network can be used for any
        dimmensions.
        TODO:
            what if the input graph already hold edge feature?
        Args:
            graph with any features
        Output:
            graph without node's features but with edge features
        """

