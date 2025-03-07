from .gnnconv import GATConv, GCNLayer, GINConv
from .layers import PairNorm
from .utils import *
from dgl.base import DGLError
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
import torch as th
linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}

class GIN(nn.Module):
    def __init__(self,
                 args,):
        super(GIN, self).__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GIN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            lin = torch.nn.Linear(dims[l], dims[l+1])
            self.gat_layers.append(GINConv(lin, 'sum'))


    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GCN_original(nn.Module):
    def __init__(self,
                 args):
        super(GCN, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))

    def forward(self, g, features):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rate):
        ctx.rate = rate
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input, rate):
        return GradReverse.apply(input, rate)

class DistingushModel(nn.Module):
    def __init__(self, IB_dim, dropout):
        super(DistingushModel, self).__init__()
        self.layer1 = GRL()
        self.layer2 = nn.Sequential(
            nn.Linear(IB_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self, h, rate=1.0):
        x1 = self.layer1(h, rate)
        output = self.layer2(x1)
        return output
    
    def reset_params(self):
        for layer in self.layer2:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class VIB(nn.Module):
    def __init__(self, dim,IB_dim, dropout):
        super(VIB, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, IB_dim)
            )
        
        self.IB_dim = IB_dim
        
    def forward(self, h):
        output = self.layer(h)
        return output
    

class Decoder(nn.Module):
    def __init__(self, dim, output_dim, dropout):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, output_dim)
            )
        
    def forward(self, h):
        output = F.relu(self.layer(h))
        return output

class GCN(nn.Module):
    def __init__(self,
                 args):
        super(GCN, self).__init__()
        self.IB_dim = int(args.GCN_args['h_dims'][-1])
        dims = [args.d_data] + args.GCN_args['h_dims'][:-1] + [self.IB_dim * 2] + [args.n_cls] # [d_data, 512, 70]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-2):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))
        self.gat_layers.append(GCNLayer(self.IB_dim, dims[-1]))

        
        self.distingush_model = DistingushModel(self.IB_dim, self.dropout)
        self.decoder = Decoder(self.IB_dim, args.d_data, self.dropout)


    def forward(self, g, features, variantion:bool=False):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        if variantion:
            mu = h[:,:self.IB_dim]
            std = F.softplus(h[:,self.IB_dim:self.IB_dim*2], beta=10)
            eps =torch.Tensor(std.size()).normal_().to(std.device)
            h = mu + eps * std
        else: 
            h = h[:, :self.IB_dim]
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e

        if variantion:
            return logits, e_list, mu, std
        else:
            return logits, e_list
        

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = h[:, :self.IB_dim]
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GAT(nn.Module):
    def __init__(self,
                 args,
                 heads,
                 activation):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = args.GAT_args['num_layers']
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(
            args.d_data, args.GAT_args['num_hidden'], heads[0],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], False, None))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, args.GAT_args['num_layers']):
            self.gat_layers.append(GATConv(
                args.GAT_args['num_hidden'] * heads[l-1], args.GAT_args['num_hidden'], heads[l],
                args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], self.activation))
            self.norm_layers.append(PairNorm())

        self.gat_layers.append(GATConv(
            args.GAT_args['num_hidden'] * heads[-2], args.n_cls, heads[-1],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], None))

    def forward(self, g, inputs, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h)
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class SGC_Agg(nn.Module):
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

    def forward_batch(self, blocks, feat):
        if self._k != len(blocks):
            raise DGLError('The depth of the dataloader sampler is incompatible with the depth of SGC')
        for block in blocks:
            with block.local_scope():
                if not self._allow_zero_in_degree:
                    if (block.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

                if self._cached_h is not None:
                    feat = self._cached_h
                else:
                    # compute normalization
                    degs = block.out_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    block.srcdata['h'] = feat
                    block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = block.dstdata.pop('h')
                    degs = block.in_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    feat = feat * norm

        with blocks[-1].local_scope():
            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        return feat

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        linear_layer = linear_choices[args.SGC_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.SGC_args['batch_norm']
        self.dropout = args.SGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = SGC_Agg(k=args.SGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.SGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.SGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.SGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.SGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.SGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, x, twp=False, tasks=None):
        e_list = []
        x = self.neighbor_agg.forward_batch(blocks, x)
        logits, e = self.feat_trans(blocks[0], x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()
