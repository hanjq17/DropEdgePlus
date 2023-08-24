from layers import *
from torch.nn.parameter import Parameter
from typing import *


device = torch.device("cuda:0")


class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 withbias=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param withbias: using bias in layers.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout
        self.midlayer_output = None

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop, bias=withbias,
                                       shared=None)
        baseblockinput = nhid

        outactivation = lambda x: x
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop,
                                        bias=withbias, shared=None)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        assert nhidlayer == 1
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 withbias=withbias,
                                 shared=None,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we do not need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop,
                                        bias=withbias, shared=None)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj: Union[torch.Tensor, List[torch.Tensor]]):
        # input
        # check whether adj is compatible with layer_num
        if isinstance(adj, list):
            assert len(adj) == (self.midlayer[0].model.nhiddenlayer if self.midlayer[0].model is not None
                                else len(self.midlayer[0].midlayers)) + 2

        if isinstance(adj, list):
            if self.mixmode:
                x = self.ingc(fea, adj[0].cpu())
            else:
                x = self.ingc(fea, adj[0])
        else:
            if self.mixmode:
                x = self.ingc(fea, adj.cpu())
            else:
                x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # currenly, only works when there's only one block
        assert len(self.midlayer) == 1

        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, (adj if not isinstance(adj, list) else adj[1: -1]))

        self.midlayer_output = x
        x = self.outgc(x, (adj if not isinstance(adj, list) else adj[-1]))
        x = F.log_softmax(x, dim=1)
        return x


class Aggr_layer(nn.Module):

    def __init__(self, in_features, out_features, alpha):
        super(Aggr_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, input_emb):
        x = torch.spmm(adj, input)
        return (1 - self.alpha) * x + self.alpha * input_emb


class New_APPNP(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, alpha):
        super(New_APPNP, self).__init__()
        self.midlayers = nn.ModuleList()
        for _ in range(nlayers):
            self.midlayers.append(Aggr_layer(nhid, nhid, alpha))
        self.input_layer, self.output_layer = nn.Linear(nfeat, nhid), nn.Linear(nhid, nclass)
        self.alpha = alpha
        self.activation = F.relu
        self.dropout = dropout

    def forward(self, x, adj, **kwargs):
        x = self.activation(self.input_layer(x))
        collector = [x]
        for _ in range(len(self.midlayers)):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation(self.midlayers[_](x, adj if not isinstance(adj, list) else adj[_], collector[0]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
