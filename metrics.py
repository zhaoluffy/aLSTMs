import argparse, os, pdb, sys, time
import numpy as np
import copy
import glob
import subprocess
from multiprocessing import Process, Queue, Manager
from collections import OrderedDict

import data_engine
from cocoeval import COCOScorer
import utils
    
MAXLEN = 50
manager = Manager()


def update_params(shared_params, model_params):
    for kk, vv in model_params.iteritems():
        shared_params[kk] = vv
    shared_params['id'] = shared_params['id'] + 1


def build_sample_pairs(samples, vidIDs):
    D = OrderedDict()
    for sample, vidID in zip(samples, vidIDs):
        D[vidID] = [{'image_id': vidID, 'caption': sample}]
    return D


def score_with_cocoeval(samples_valid, samples_test, engine):
    scorer = COCOScorer()
    if samples_valid:
        gts_valid = OrderedDict()
        for vidID in engine.valid_ids:
            gts_valid[vidID] = engine.CAP[vidID]
        valid_score = scorer.score(gts_valid, samples_valid, engine.valid_ids)
    else:
        valid_score = None
    if samples_test:
        gts_test = OrderedDict()
        for vidID in engine.test_ids:
            gts_test[vidID] = engine.CAP[vidID]
        test_score = scorer.score(gts_test, samples_test, engine.test_ids)
    else:
        test_score = None
    return valid_score, test_score


def generate_sample_gpu_single_process(
        model_type, model_archive, options, engine, model,
        f_init, f_next,
        save_dir='./samples', beam=5,
        whichset='both'):
    
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(engine.ix_word[1]
                          if w > len(engine.ix_word) else engine.ix_word[w])
            capsw.append(' '.join(ww))
        return capsw
    
    def sample(whichset):
        samples = []
        ctxs, ctx_masks = engine.prepare_data_for_blue(whichset)
        for i, ctx, ctx_mask in zip(range(len(ctxs)), ctxs, ctx_masks):
            print 'sampling %d/%d'%(i,len(ctxs))
            sample, score, _, _ = model.gen_sample(None, f_init, f_next,
                                                   ctx, ctx_mask,None,
                                                   beam, maxlen=MAXLEN)
            
            sidx = np.argmin(score)
            sample = sample[sidx]
            #print _seqs2words([sample])[0]
            samples.append(sample)
        samples = _seqs2words(samples)
        return samples

    if whichset == 'valid' or whichset == 'both':
        print 'Valid Set...',
        samples_valid = sample('valid')
        with open(save_dir+'/valid_samples.txt', 'w') as f:
            print >>f, '\n'.join(samples_valid)
    if whichset == 'test' or whichset == 'both':
        print 'Test Set...',
        samples_test = sample('test')
        with open(save_dir+'/test_samples.txt', 'w') as f:
            print >>f, '\n'.join(samples_test)

    if samples_valid:
        samples_valid = build_sample_pairs(samples_valid, engine.valid_ids)
    if samples_test:
        samples_test = build_sample_pairs(samples_test, engine.test_ids)
    return samples_valid, samples_test


def compute_score(
        model_type, model_archive, options, engine, save_dir,
        beam, n_process,
        whichset='both', on_cpu=True,
        processes=None, queue=None, rqueue=None, shared_params=None,
        one_time=False, metric=None,
        f_init=None, f_next=None, model=None):

    assert metric != 'perplexity'
    if on_cpu:
        raise NotImplementedError()
    else:
        assert model is not None
        samples_valid, samples_test = generate_sample_gpu_single_process(
            model_type, model_archive,options,
            engine, model, f_init, f_next,
            save_dir=save_dir,
            beam=beam,
            whichset=whichset)
        
    valid_score, test_score = score_with_cocoeval(samples_valid, samples_test, engine)
    scores_final = {}
    scores_final['valid'] = valid_score
    scores_final['test'] = test_score
    
    if one_time:
        return scores_final
    
    return scores_final, processes, queue, rqueue, shared_params    


def test_cocoeval():
    engine = data_engine.Movie2Caption('attention', 'youtube2text',
                                       video_feature='googlenet',
                                       mb_size_train=20,
                                       mb_size_test=20,
                                       maxlen=50, n_words=20000,
                                       n_frames=20, outof=None)
    samples_valid = utils.load_txt_file('./test/valid_samples.txt')
    samples_test = utils.load_txt_file('./test/test_samples.txt')
    samples_valid = [sample.strip() for sample in samples_valid]
    samples_test = [sample.strip() for sample in samples_test]

    samples_valid = build_sample_pairs(samples_valid, engine.valid_ids)
    samples_test = build_sample_pairs(samples_test, engine.test_ids)
    valid_score, test_score = score_with_cocoeval(samples_valid, samples_test, engine)
    print valid_score, test_score

if __name__ == '__main__':
    test_cocoeval()
