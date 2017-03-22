import numpy as np
import theano
import theano.tensor as T
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs


class Layers(object):

    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('self.param_init_lstm', 'self.lstm_layer'),
            'lstm_cond': ('self.param_init_lstm_cond', 'self.lstm_cond_layer'),
            }
        self.rng_numpy, self.rng_theano = get_two_rngs()

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return eval(fns[0]), eval(fns[1])

    def dropout_layer(self, state_before, use_noise, trng):

        proj = T.switch(use_noise,
                             state_before *
                             trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                             state_before * 0.5)
        return proj

    def fflayer(self, tparams, state_below, activ='lambda x: T.tanh(x)', prefix='ff', **kwargs):

        return eval(activ)(T.dot(state_below, tparams[_p(prefix,'W')])+
                           tparams[_p(prefix,'b')])

    def param_init_fflayer(self, params, nin, nout, prefix=None):
        assert prefix is not None
        params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
        return params

    def param_init_lstm(self, params, nin, dim, prefix='lstm'):
        assert prefix is not None
        # Stack the weight matricies for faster dot prods
        W = np.concatenate([norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim)], axis=1)
        params[_p(prefix, 'W')] = W
        U = np.concatenate([ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U
        params[_p(prefix, 'b')] = np.zeros((4 * dim,)).astype('float32')

        return params

    def lstm_layer(self, tparams, state_below, mask=None, init_state=None, init_memory=None,
                   one_step=False, prefix='lstm', **kwargs):

        # state_below (t, m, dim_word), or (m, dim_word) in sampling

        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'

        n_steps = state_below.shape[0]
        dim = tparams[_p(prefix, 'U')].shape[0]

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
            if init_state is None:
                init_state = T.alloc(0., n_samples, dim)
            if init_memory is None:
                init_memory = T.alloc(0., n_samples, dim)
        else:
            n_samples = 1
            if init_state is None:
                init_state = T.alloc(0., dim)
            if init_memory is None:
                init_memory = T.alloc(0., dim)

        if mask is None:
            mask = T.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]

        U = tparams[_p(prefix, 'U')]
        b = tparams[_p(prefix, 'b')]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, U)
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, dim))
            f = T.nnet.sigmoid(_slice(preact, 1, dim))
            o = T.nnet.sigmoid(_slice(preact, 2, dim))
            c = T.tanh(_slice(preact, 3, dim))

            c = f * c_ + i * c
            h = o * T.tanh(c)
            if m_.ndim == 0:
                # when using this for minibatchsize=1
                h = m_ * h + (1. - m_) * h_
                c = m_ * c + (1. - m_) * c_
            else:
                h = m_[:, None] * h + (1. - m_)[:, None] * h_
                c = m_[:, None] * c + (1. - m_)[:, None] * c_
            return h, c

        state_below = T.dot(
            state_below, tparams[_p(prefix, 'W')]) + b

        if one_step:
            rval = _step(mask, state_below, init_state, init_memory)
        else:
            rval, updates = theano.scan(_step,
                                        sequences=[mask, state_below],
                                        outputs_info=[init_state, init_memory],
                                        name=_p(prefix,'_layers'),
                                        n_steps=n_steps,
                                        profile=False
                                        )
        return rval

    # Conditional LSTM layer with Attention
    def param_init_lstm_cond(self, options, params, nin, dim, dimctx,
                             prefix='lstm_cond'):

        # input to LSTM
        W = np.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix, 'W')] = W

        # LSTM to LSTM
        U = np.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U

        # bias to LSTM
        params[_p(prefix, 'b')] = np.zeros((4 * dim,)).astype('float32')

        # context to LSTM
        Wc = norm_weight(dimctx,dim*4)
        params[_p(prefix, 'Wc')] = Wc

        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix, 'Wc_att')] = Wc_att

        # attention: LSTM -> hidden
        Wd_att = norm_weight(dim,dimctx)
        params[_p(prefix, 'Wd_att')] = Wd_att

        # attention: hidden bias
        b_att = np.zeros((dimctx,)).astype('float32')
        params[_p(prefix, 'b_att')] = b_att

        # attention:
        U_att = norm_weight(dimctx, 1)
        params[_p(prefix, 'U_att')] = U_att
        c_att = np.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_tt')] = c_att

        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel
            b_sel = np.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel

        return params

    def lstm_cond_layer(self, options, tparams, state_below,
                        mask=None, context=None, one_step=False,
                        init_state=None, init_memory=None,
                        trng=None, use_noise=None, prefix='lstm', **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (f, dim_word) in sampling
        # init_memory, init_state (m, dim)
        assert context, 'Context must be provided'

        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        # mask
        if mask is None:
            mask = T.alloc(1., state_below.shape[0], 1)

        dim = tparams[_p(prefix, 'U')].shape[0]

        # initial/previous state
        if init_state is None:
            init_state = T.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory is None:
            init_memory = T.alloc(0., n_samples, dim)

        # projected context
        pctx_ = T.dot(context, tparams[_p(prefix,'Wc_att')]) + \
                tparams[_p(prefix, 'b_att')]
        if one_step:
            # tensor.dot will remove broadcasting dim
            pctx_ = T.addbroadcast(pctx_, 0)

        # projected x
        state_below = T.dot(state_below, tparams[_p(prefix, 'W')]) + \
                      tparams[_p(prefix, 'b')]

        Wd_att = tparams[_p(prefix, 'Wd_att')]
        U_att = tparams[_p(prefix, 'U_att')]
        c_att = tparams[_p(prefix, 'c_tt')]
        if options['selector']:
            W_sel = tparams[_p(prefix, 'W_sel')]
            b_sel = tparams[_p(prefix, 'b_sel')]
        else:
            W_sel = T.alloc(0., 1)
            b_sel = T.alloc(0., 1)
        U = tparams[_p(prefix, 'U')]
        Wc = tparams[_p(prefix, 'Wc')]

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_, x_, # sequences
                  h_, c_, a_, ctx_, # outputs_info
                  dp_=None # non_sequences
                  ):
            # attention
            pstate_ = T.dot(h_, Wd_att)
            pstate_ = pctx_ + pstate_[:,None,:]
            pstate_ = tanh(pstate_)

            alpha = T.dot(pstate_, U_att)+c_att
            alpha_shp = alpha.shape
            alpha = T.nnet.softmax(alpha.reshape([alpha_shp[0], alpha_shp[1]])) # softmax
            ctx_ = (context * alpha[:, :, None]).sum(1) # (m, ctx_dim)
            if options['selector']:
                sel_ = T.nnet.sigmoid(T.dot(h_, W_sel) + b_sel)
                sel_ = sel_.reshape([sel_.shape[0]])
                ctx_ = sel_[:,None] * ctx_
            preact = T.dot(h_, U)
            preact += x_
            preact += T.dot(ctx_, Wc)

            i = _slice(preact, 0, dim)
            f = _slice(preact, 1, dim)
            o = _slice(preact, 2, dim)
            if options['use_dropout']:
                i *= _slice(dp_, 0, dim)
                f *= _slice(dp_, 1, dim)
                o *= _slice(dp_, 2, dim)
            i = T.nnet.sigmoid(i)
            f = T.nnet.sigmoid(f)
            o = T.nnet.sigmoid(o)
            c = T.tanh(_slice(preact, 3, dim))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            rval = [h, c, alpha, ctx_]
            return rval
        if options['use_dropout']:
            _step0 = lambda m_, x_, dp_, h_, c_, a_, ctx_: \
                _step(m_, x_, h_, c_, a_, ctx_, dp_)
            dp_shape = state_below.shape
            if one_step:
                dp_mask = T.switch(use_noise,
                                   trng.binomial((dp_shape[0], 3*dim),
                                                 p=0.5, n=1, dtype=state_below.dtype),
                                   T.alloc(0.5, dp_shape[0], 3 * dim))
            else:
                dp_mask = T.switch(use_noise,
                                   trng.binomial((dp_shape[0], dp_shape[1], 3*dim),
                                                 p=0.5, n=1, dtype=state_below.dtype),
                                   T.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
        else:
            _step0 = lambda m_, x_, h_, c_, a_, ctx_: \
                _step(m_, x_, h_, c_, a_, ctx_)

        if one_step:
            if options['use_dropout']:
                rval = _step0(mask, state_below, dp_mask,
                              init_state, init_memory, None, None)
            else:
                rval = _step0(mask, state_below,
                              init_state, init_memory, None, None)
        else:
            seqs = [mask, state_below]
            if options['use_dropout']:
                seqs += [dp_mask]
            rval, updates = theano.scan(_step0,
                                        sequences=seqs,
                                        outputs_info = [init_state,
                                                        init_memory,
                                                        T.alloc(0., n_samples, pctx_.shape[1]),
                                                        T.alloc(0., n_samples, context.shape[2])
                                                        ],
                                        name=_p(prefix, '_layers'),
                                        n_steps=nsteps, profile=False)

        return rval
