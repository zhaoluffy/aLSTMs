import cPickle as pkl
import os
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import warnings

def _p(pp, name):
    return '%s_%s'%(pp, name)


def get_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = np.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)
    return rng_numpy, rng_theano

rng_numpy, rng_theano = get_two_rngs()


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = T.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

'''
Theano shared variables require GPUs, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy
dictionary
'''


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are
    orthogonal.
    """
    W = rng_numpy.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_numpy.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return T.tanh(x)


def rectifier(x):
    return T.maximum(0., x)


def linear(x):
    return x


# load parameters
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]
    return params


def grad_nan_report(grads, tparams):
    np.set_printoptions(precision=3)
    D = OrderedDict()
    i = 0
    NaN_keys = []
    magnitude = []
    assert len(grads) == len(tparams)
    for k, v in tparams.iteritems():
        grad = grads[i]
        magnitude.append(np.abs(grad).mean())
        if np.isnan(grad.sum()):
            NaN_keys.append(k)
        D[k] = grad
        i += 1
    return D, NaN_keys


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = pkl.load(f)
    finally:
        f.close()
    return rval


def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    finally:
        f.close()


def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = np.split(np.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = np.split(np.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [np.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx


def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory


def flatten_list_of_list(l):
    # l is a list of list
    return [item for sublist in l for item in sublist]


def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines


