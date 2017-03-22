from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
import time
from layers import Layers
import data_engine
import metrics
from optimizers import *

from predict import *


def validate_options(options):
    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')
    return options


class Attention(object):
    def __init__(self, channel=None):
        self.rng_numpy, self.rng_theano = get_two_rngs()
        self.layers = Layers()
        self.predict = Predict()
        self.channel = channel

    def load_params(self, path, params):
        # load params from disk
        pp = np.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive'%kk)
            params[kk] = pp[kk]

        return params

    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        ctx_dim = options['ctx_dim']
        if options['isGlobal']:
            # Embedding
            params = self.layers.get_layer('ff')[0](
                     params, nin=options['encoder_dim'], nout=options['dim_word'], prefix='ff_embed')

        # encoder: LSTMz
        params = self.layers.get_layer('lstm')[0](params, nin=ctx_dim,
                                                  dim=options['encoder_dim'], prefix='en_lstm')

        # init_state, init_cell
        params = self.layers.get_layer('ff')[0](params, nin=options['encoder_dim'], nout=options['dim'],
                                                prefix='ff_state')
        params = self.layers.get_layer('ff')[0](params, nin=options['encoder_dim'], nout=options['dim'],
                                                prefix='ff_memory')
        # decoder: LSTM
        params = self.layers.get_layer('lstm_cond')[0](options, params, nin=options['dim_word'],
                                                       dim=options['dim'], dimctx=ctx_dim, prefix='decoder')

        # readout
        params = self.layers.get_layer('ff')[0](params, nin=options['dim'], nout=options['dim_word'],
                                                prefix='ff_logit_lstm')

        if options['ctx2out']:
            params = self.layers.get_layer('ff')[0](params, nin=ctx_dim, nout=options['dim_word'],
                                                    prefix='ff_logit_ctx')

        params = self.layers.get_layer('ff')[0](params, nin=options['dim_word'], nout=options['n_words'],
                                                prefix='ff_logit')
        return params

    def build_model(self, tparams, options):
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype='float32')
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')
        mask_ctx = tensor.matrix('mask_ctx', dtype='float32')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        emb = tparams['Wemb'][x.flatten()].reshape(
                [n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        ctx_ = ctx
        # counts = mask_ctx.sum(-1).dimshuffle(0,'x')
        # ctx_mean = ctx_.sum(1)/counts

        # encoder
        en_lstm = self.layers.get_layer('lstm')[1](tparams, ctx_.dimshuffle(1, 0, 2),
                                                   mask=mask_ctx.dimshuffle(1, 0), prefix='en_lstm')

        if options['isGlobal']:
            # embedding visual context
            ctx_emb = self.layers.get_layer('ff')[1](tparams, en_lstm[0][-1, :, :],
                                                     activ='tanh', prefix='ff_embed')

        # initial state/cell
        init_state = self.layers.get_layer('ff')[1](tparams, en_lstm[0][-1, :, :],
                                                    activ='tanh', prefix='ff_state')
        init_memory = self.layers.get_layer('ff')[1](tparams, en_lstm[0][-1, :, :],
                                                     activ='tanh', prefix='ff_memory')
        # decoder
        mu_lstm = self.layers.get_layer('lstm_cond')[1](options, tparams, emb,
                                                        mask=mask, context=ctx_,
                                                        one_step=False,
                                                        init_state=init_state,
                                                        init_memory=init_memory,
                                                        trng=trng,
                                                        use_noise=use_noise,
                                                        prefix='decoder')

        proj_h = mu_lstm[0]
        alphas = mu_lstm[2]
        ctxs = mu_lstm[3]
        if options['use_dropout']:
            proj_h = self.layers.dropout_layer(proj_h, use_noise, trng)

        # compute word probabilities
        logit = self.layers.get_layer('ff')[1](tparams, proj_h, activ='linear',
                                               prefix='ff_logit_lstm')
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            logit += self.layers.get_layer('ff')[1](tparams, ctxs, activ='linear',
                                                    prefix='ff_logit_ctx')
        logit = tanh(logit)
        if options['use_dropout']:
            logit = self.layers.dropout_layer(logit, use_noise, trng)

        # (t,m,n_words)
        logit = self.layers.get_layer('ff')[1](tparams, logit,
                                               activ='linear', prefix='ff_logit')
        logit_shp = logit.shape
        # (t*m, n_words)
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))

        # cost
        x_flat = x.flatten() # (t*m,)
        cost = -tensor.log(probs[T.arange(x_flat.shape[0]), x_flat] + 1e-8)
        cost = cost.reshape([x.shape[0], x.shape[1]])
        cost = (cost * mask).sum(0)

        if options['isGlobal']:
            # my cost
            word_counts = mask.sum(0).dimshuffle(0, 'x')
            emb_mean = emb.sum(0)/word_counts
            my_loss = T.sqrt(T.sum((emb_mean-ctx_emb)**2))
            lamb = 0.1
            # end my cost
            cost = (1-lamb)*cost + lamb*my_loss
            extra = [probs, alphas, my_loss]
        else:
            extra = [probs, alphas]
        return trng, use_noise, x, mask, ctx, mask_ctx, alphas, cost, extra

    def pred_probs(self, whichset, f_log_probs, verbose=True):

        probs = []
        n_done = 0
        NLL = []
        L = []
        if whichset == 'train':
            tags = self.engine.train
            iterator = self.engine.kf_train
        elif whichset == 'valid':
            tags = self.engine.valid
            iterator = self.engine.kf_valid
        elif whichset == 'test':
            tags = self.engine.test
            iterator = self.engine.kf_test
        else:
            raise NotImplementedError()
        n_samples = np.sum([len(index) for index in iterator])
        for index in iterator:
            tag = [tags[i] for i in index]
            x, mask, ctx, ctx_mask = data_engine.prepare_data(
                self.engine, tag)
            pred_probs = f_log_probs(x, mask, ctx, ctx_mask)
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(tag)
            if verbose:
                sys.stdout.write('\rComputing LL on %d/%d examples'%(
                             n_done, n_samples))
                sys.stdout.flush()
        print
        probs = flatten_list_of_list(probs)
        NLL = flatten_list_of_list(NLL)
        L = flatten_list_of_list(L)
        perp = 2**(np.sum(NLL) / np.sum(L) / np.log(2))
        return -1 * np.mean(probs), perp

    def train(self,
              random_seed=1234,
              reload_=False,
              verbose=True,
              debug=True,
              save_model_dir='',
              from_dir=None,
              # dataset
              dataset='youtube2text',
              video_feature='googlenet',
              K=10,
              OutOf=240,
              # network
              dim_word=256, # word vector dimensionality
              ctx_dim=-1, # context vector dimensionality, auto set
              dim=1000, # the number of LSTM units
              n_layers_out=1,
              n_layers_init=1,
              encoder='none',
              encoder_dim=100,
              prev2out=False,
              ctx2out=False,
              selector=False,
              n_words=100000,
              maxlen=100, # maximum length of the description
              use_dropout=False,
              isGlobal=False,
              # training
              patience=10,
              max_epochs=5000,
              decay_c=0.,
              alpha_c=0.,
              alpha_entropy_r=0.,
              lrate=0.01,
              optimizer='adadelta',
              clip_c=2.,
              # minibatch
              batch_size = 64,
              valid_batch_size = 64,
              dispFreq=100,
              validFreq=10,
              saveFreq=10, # save the parameters after every saveFreq updates
              sampleFreq=10, # generate some samples after every sampleFreq updates
              # metric
              metric='blue'
              ):
        self.rng_numpy, self.rng_theano = get_two_rngs()

        model_options = locals().copy()
        if 'self' in model_options:
            del model_options['self']
        model_options = validate_options(model_options)
        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
            pkl.dump(model_options, f)

        print 'Loading data'
        self.engine = data_engine.Movie2Caption('attention', dataset,
                                           video_feature,
                                           batch_size, valid_batch_size,
                                           maxlen, n_words,
                                           K, OutOf)
        model_options['ctx_dim'] = self.engine.ctx_dim
        model_options['n_words'] = self.engine.n_words

        print 'init params'
        t0 = time.time()
        params = self.init_params(model_options)

        # reloading
        if reload_:
            model_saved = from_dir+'/model_best_so_far.npz'
            assert os.path.isfile(model_saved)
            print "Reloading model params..."
            params = load_params(model_saved, params)

        tparams = init_tparams(params)
        if verbose:
            print tparams.keys

        trng, use_noise, x, mask, ctx, mask_ctx, alphas, cost, extra = \
            self.build_model(tparams, model_options)

        print 'buliding sampler'
        f_init, f_next = self.predict.build_sampler(self.layers, tparams, model_options, use_noise, trng)
        # before any regularizer
        print 'building f_log_probs'
        f_log_probs = theano.function([x, mask, ctx, mask_ctx], -cost,
                                      profile=False, on_unused_input='ignore')

        cost = cost.mean()
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        if alpha_c > 0.:
            alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
            alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
            cost += alpha_reg

        if alpha_entropy_r > 0:
            alpha_entropy_r = theano.shared(np.float32(alpha_entropy_r),
                                            name='alpha_entropy_r')
            alpha_reg_2 = alpha_entropy_r * (-tensor.sum(alphas *
                        tensor.log(alphas+1e-8),axis=-1)).sum(0).mean()
            cost += alpha_reg_2
        else:
            alpha_reg_2 = tensor.zeros_like(cost)
        print 'building f_alpha'
        f_alpha = theano.function([x, mask, ctx, mask_ctx],
                                  [alphas, alpha_reg_2],
                                  name='f_alpha',
                                  on_unused_input='ignore')

        print 'compute grad'
        grads = tensor.grad(cost, wrt=itemlist(tparams))
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c**2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        lr = tensor.scalar(name='lr')
        print 'build train fns'
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads,
                                                  [x, mask, ctx, mask_ctx], cost,
                                                  extra + grads)

        print 'compilation took %.4f sec'%(time.time()-t0)
        print 'Optimization'

        history_errs = []
        # reload history
        if reload_:
            print 'loading history error...'
            history_errs = np.load(
                from_dir+'model_best_so_far.npz')['history_errs'].tolist()

        bad_counter = 0

        processes = None
        queue = None
        rqueue = None
        shared_params = None

        uidx = 0
        uidx_best_blue = 0
        uidx_best_valid_err = 0
        estop = False
        best_p = unzip(tparams)
        best_blue_valid = 0
        best_valid_err = 999
        alphas_ratio = []
        for eidx in xrange(max_epochs):
            n_samples = 0
            train_costs = []
            grads_record = []
            print 'Epoch ', eidx
            for idx in self.engine.kf_train:
                tags = [self.engine.train[index] for index in idx]
                n_samples += len(tags)
                uidx += 1
                use_noise.set_value(1.)

                pd_start = time.time()
                x, mask, ctx, ctx_mask = data_engine.prepare_data(
                    self.engine, tags)
                pd_duration = time.time() - pd_start
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue

                ud_start = time.time()
                rvals = f_grad_shared(x, mask, ctx, ctx_mask)
                cost = rvals[0]
                probs = rvals[1]
                alphas = rvals[2]
                if isGlobal:
                    my_loss = rvals[3]
                    grads = rvals[4:]
                else:
                    grads = rvals[3:]
                grads, NaN_keys = grad_nan_report(grads, tparams)
                if len(grads_record) >= 5:
                    del grads_record[0]
                grads_record.append(grads)
                if NaN_keys != []:
                    print 'grads contain NaN'
                    import pdb; pdb.set_trace()
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected in cost'
                    import pdb; pdb.set_trace()
                # update params
                f_update(lrate)
                ud_duration = time.time() - ud_start

                if eidx == 0:
                    train_error = cost
                else:
                    train_error = train_error * 0.95 + cost * 0.05
                train_costs.append(cost)

                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Train cost mean so far', \
                      train_error, 'fetching data time spent (sec)', np.round(pd_duration,3), \
                      'update time spent (sec)', np.round(ud_duration,3)
                    alphas,reg = f_alpha(x,mask,ctx,ctx_mask)
                    print 'alpha ratio %.3f, reg %.3f' % (
                        alphas.min(-1).mean() / (alphas.max(-1)).mean(), reg)
                    if isGlobal:
                        print ' my loss %.3f' % (my_loss)
                if np.mod(uidx, saveFreq) == 0:
                    pass

                if np.mod(uidx, sampleFreq) == 0:
                    use_noise.set_value(0.)
                    print '------------- sampling from train ----------'
                    self.predict.sample_execute(self.engine, model_options, tparams,
                                                f_init, f_next, x, ctx, ctx_mask, trng)

                    print '------------- sampling from valid ----------'
                    idx = self.engine.kf_valid[np.random.randint(1, len(self.engine.kf_valid) - 1)]
                    tags = [self.engine.valid[index] for index in idx]
                    x_s, mask_s, ctx_s, mask_ctx_s = data_engine.prepare_data(self.engine, tags)
                    self.predict.sample_execute(self.engine, model_options, tparams,
                                                f_init, f_next, x_s, ctx_s, mask_ctx_s, trng)
                    # end of sample

                if validFreq != -1 and np.mod(uidx, validFreq) == 0:
                    t0_valid = time.time()
                    alphas,_ = f_alpha(x, mask, ctx, ctx_mask)
                    ratio = alphas.min(-1).mean()/(alphas.max(-1)).mean()
                    alphas_ratio.append(ratio)
                    np.savetxt(save_model_dir+'alpha_ratio.txt',alphas_ratio)

                    current_params = unzip(tparams)
                    np.savez(save_model_dir+'model_current.npz',
                             history_errs=history_errs, **current_params)

                    use_noise.set_value(0.)
                    train_err = -1
                    train_perp = -1
                    valid_err = -1
                    valid_perp = -1
                    test_err = -1
                    test_perp = -1
                    if not debug:
                        # first compute train cost
                        if 0:
                            print 'computing cost on trainset'
                            train_err, train_perp = self.pred_probs(
                                    'train', f_log_probs,
                                    verbose=model_options['verbose'])
                        else:
                            train_err = 0.
                            train_perp = 0.
                        if 0:
                            print 'validating...'
                            valid_err, valid_perp = self.pred_probs(
                                'valid', f_log_probs,
                                verbose=model_options['verbose'],
                                )
                        else:
                            valid_err = 0.
                            valid_perp = 0.
                        if 0:
                            print 'testing...'
                            test_err, test_perp = self.pred_probs(
                                'test', f_log_probs,
                                verbose=model_options['verbose']
                                )
                        else:
                            test_err = 0.
                            test_perp = 0.

                    mean_ranking = 0
                    blue_t0 = time.time()
                    scores, processes, queue, rqueue, shared_params = \
                        metrics.compute_score(model_type='attention',
                                              model_archive=current_params,
                                              options=model_options,
                                              engine=self.engine,
                                              save_dir=save_model_dir,
                                              beam=5, n_process=5,
                                              whichset='both',
                                              on_cpu=False,
                                              processes=processes, queue=queue, rqueue=rqueue,
                                              shared_params=shared_params, metric=metric,
                                              one_time=False,
                                              f_init=f_init, f_next=f_next, model=self.predict
                                              )

                    valid_B1 = scores['valid']['Bleu_1']
                    valid_B2 = scores['valid']['Bleu_2']
                    valid_B3 = scores['valid']['Bleu_3']
                    valid_B4 = scores['valid']['Bleu_4']
                    valid_Rouge = scores['valid']['ROUGE_L']
                    valid_Cider = scores['valid']['CIDEr']
                    valid_meteor = scores['valid']['METEOR']
                    test_B1 = scores['test']['Bleu_1']
                    test_B2 = scores['test']['Bleu_2']
                    test_B3 = scores['test']['Bleu_3']
                    test_B4 = scores['test']['Bleu_4']
                    test_Rouge = scores['test']['ROUGE_L']
                    test_Cider = scores['test']['CIDEr']
                    test_meteor = scores['test']['METEOR']
                    print 'computing meteor/blue score used %.4f sec, '\
                          'blue score: %.1f, meteor score: %.1f'%(
                    time.time()-blue_t0, valid_B4, valid_meteor)
                    history_errs.append([eidx, uidx, train_err, train_perp,
                                         valid_perp, test_perp,
                                         valid_err, test_err,
                                         valid_B1, valid_B2, valid_B3,
                                         valid_B4, valid_meteor, valid_Rouge, valid_Cider,
                                         test_B1, test_B2, test_B3,
                                         test_B4, test_meteor, test_Rouge, test_Cider])
                    np.savetxt(save_model_dir+'train_valid_test.txt',
                                  history_errs, fmt='%.3f')
                    print 'save validation results to %s'%save_model_dir
                    # save best model according to the best blue or meteor
                    if len(history_errs) > 1 and valid_B4 > np.array(history_errs)[:-1,11].max():
                        print 'Saving to %s...'%save_model_dir,
                        np.savez(
                            save_model_dir+'model_best_blue_or_meteor.npz',
                            history_errs=history_errs, **best_p)
                    if len(history_errs) > 1 and valid_err < np.array(history_errs)[:-1,6].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                        best_valid_err = valid_err
                        uidx_best_valid_err = uidx

                        print 'Saving to %s...'%save_model_dir,
                        np.savez(
                            save_model_dir+'model_best_so_far.npz',
                            history_errs=history_errs, **best_p)
                        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
                            pkl.dump(model_options, f)
                        print 'Done'
                    elif len(history_errs) > 1 and valid_err >= np.array(history_errs)[:-1,6].min():
                        bad_counter += 1
                        print 'history best ',np.array(history_errs)[:,6].min()
                        print 'bad_counter ',bad_counter
                        print 'patience ',patience
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if self.channel:
                        self.channel.save()

                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, \
                          'best valid err so far',best_valid_err
                    print 'valid took %.2f sec'%(time.time() - t0_valid)
                    # end of validatioin
                if debug:
                    break
            if estop:
                break
            if debug:
                break

            # end for loop over minibatches
            print 'This epoch has seen %d samples, train cost %.2f'%(
                n_samples, np.mean(train_costs))
        # end for loop over epochs
        print 'Optimization ended.'
        if best_p is not None:
            zipp(best_p, tparams)

        print 'stopped at epoch %d, minibatch %d, '\
              'curent Train %.2f, current Valid %.2f, current Test %.2f '%(
               eidx, uidx, np.mean(train_err), np.mean(valid_err), np.mean(test_err))
        params = copy.copy(best_p)
        np.savez(save_model_dir+'model_best.npz',
                 train_err=train_err,
                 valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                 **params)

        if history_errs != []:
            history = np.asarray(history_errs)
            best_valid_idx = history[:,6].argmin()
            np.savetxt(save_model_dir+'train_valid_test.txt', history, fmt='%.4f')
            print 'final best exp ', history[best_valid_idx]

        return train_err, valid_err, test_err


def train_from_scratch(state, channel):
    t0 = time.time()
    print 'training an attention model'
    model = Attention(channel)
    model.train(**state.attention)
    print 'training time in total %.4f sec'%(time.time()-t0)

