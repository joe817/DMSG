import torch as th
from torch import nn
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair
from dgl.base import DGLError
import math
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
import torch.autograd as autograd

class GINConv(nn.Module):
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))
        self.learn_eps = learn_eps
        self.init_eps = init_eps

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        self.eps.fill_(self.init_eps)
        self.apply_func.reset_parameters()

    def forward(self, graph, feat):
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat)
        graph.srcdata['h'] = feat_src
        graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        graph.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = graph.edata.pop('e')

        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        return rst, elist

    def forward_batch(self, block, feat):
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat, block)

        block.srcdata['h'] = feat_src
        block.dstdata['h'] = feat_dst

        block.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * feat_dst + block.dstdata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        block.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = block.edata.pop('e')

        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        return rst, elist


def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, negative_slope=0.2):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, feat):
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        h = self.linear(feat)
        graph.ndata['h'] = h
        graph.update_all(gcn_msg, gcn_reduce)
        h = graph.ndata['h']

        graph.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = self.leaky_relu(graph.edata.pop('e'))

        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        return h, elist

    def forward_batch(self, block, feat):
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat)
        h = self.linear(feat_src)
        block.srcdata['h'] = h
        block.update_all(gcn_msg, gcn_reduce)
        h = block.dstdata['h']

        block.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = self.leaky_relu(block.edata.pop('e'))

        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        return h, elist

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 k = 1):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
            
        self.attn_l1 = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r1 = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.lrelu = nn.Sigmoid()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l1, gain=gain)
        nn.init.xavier_normal_(self.attn_r1, gain=gain)        
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        
    def forward(self, graph, feat):
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l1).sum(dim=-1).unsqueeze(-1) 
        er = (feat * self.attn_r1).sum(dim=-1).unsqueeze(-1) 
        graph.ndata.update({'ft': feat, 'el': el, 'er': er}) 
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))      
        e = self.leaky_relu(graph.edata.pop('e'))  
        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        graph.edata['a'] = self.attn_drop(e_soft)       
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) 
        rst = graph.ndata['ft'] 
        if self.activation:
            rst = self.activation(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        return rst, elist

    def forward_batch(self, block, feat):
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        h_src = h_dst = self.feat_drop(feat)
        feat_src = self.fc(h_src).view(
            -1, self._num_heads, self._out_feats)
        feat_dst = feat_src[:block.number_of_dst_nodes()]
        el = (feat_src * self.attn_l1).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r1).sum(dim=-1).unsqueeze(-1)
        block.srcdata.update({'ft': feat_src, 'el': el})
        block.dstdata.update({'er': er})
        block.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(block.edata.pop('e'))
        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        block.edata['a'] = self.attn_drop(e_soft)
        block.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = block.dstdata['ft']
        if self.activation:
            rst = self.activation(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        return rst, elist
