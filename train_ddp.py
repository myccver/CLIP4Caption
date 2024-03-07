# -*- coding: UTF-8 -*-
import os
import sys
import time
import math
import json
import uuid
import logging
import numpy as np

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm, clip_grad_norm_

from datetime import datetime
from six.moves import cPickle

from model import CrossEntropyCriterion, RewardCriterion, CaptionModel_clip
from torch.utils.data import DataLoader
import utils
import opts
from caption_dataloader import CaptionDataset
# sys.path.append("cider")
# from pyciderevalcap.cider.cider import Cider
# from pyciderevalcap.ciderD.ciderD import CiderD
#
# sys.path.append('coco-caption')
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from caption_dataloader import CaptionDataset, collate_fn_caption
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


logger = logging.getLogger(__name__)


def check_model(model, opt, infos, infos_history):
    if opt.eval_metric == 'MSRVTT':
        current_score = infos['Bleu_4'] + \
                        infos['METEOR'] + infos['ROUGE_L'] + infos['CIDEr']
    else:
        current_score = infos[opt.eval_metric]

    # write the full model checkpoint as well if we did better than ever
    if current_score >= infos['best_score']:
        infos['best_score'] = current_score
        infos['best_iter'] = infos['iter']
        infos['best_epoch'] = infos['epoch']

        logger.info('>>> Found new best [%s] score: %f, at iter: %d, epoch %d', opt.eval_metric, current_score,
                    infos['iter'], infos['epoch'])

        torch.save({'model': model.state_dict(), 'infos': infos, 'opt': opt}, opt.model_file)
        logger.info('Wrote checkpoint to: %s', opt.model_file)

    else:
        logger.info('>>> Current best [%s] score: %f, at iter %d, epoch %d', opt.eval_metric, infos['best_score'],
                    infos['best_iter'], infos['best_epoch'])

    infos_history[infos['epoch']] = infos.copy()
    with open(opt.history_file, 'w') as of:
        json.dump(infos_history, of)
    logger.info('Updated history to: %s', opt.history_file)


def train(model, criterion, optimizer,lr_scheduler, train_loader, val_loader, opt, rl_criterion=None):
    infos = {'iter': 0, 'epoch': 0, 'start_epoch': 0, 'best_score': float('-inf'), 'best_iter': 0,
             'best_epoch': opt.max_epochs}

    checkpoint_checked = False
    rl_training = False
    # seq_per_img = train_loader.get_seq_per_img()
    infos_history = {}

    if os.path.exists(opt.start_from):
        # loading the same model file at a different experiment dir
        start_from_file = os.path.join(opt.start_from, os.path.basename(opt.model_file)) if os.path.isdir(
            opt.start_from) else opt.start_from
        logger.info('Loading state from: %s', start_from_file)
        checkpoint = torch.load(start_from_file)
        model.load_state_dict(checkpoint['model'])
        infos = checkpoint['infos']
        infos['start_epoch'] = infos['epoch']
        checkpoint_checked = True  # this epoch is already checked
    else:
        logger.info('No checkpoint found! Training from the scratch')


    for epoch in range(opt.max_epochs):
        train_sampler.set_epoch(epoch)  # 设置随机种子
        t_start = time.time()
        model.train()
        for i, (feats,labels,masks) in enumerate(train_loader):
            if torch.cuda.is_available():
                feats = feats.cuda()
                # motion = motion.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
            optimizer.zero_grad()
            pred = model(feats, labels)[0]
            loss = criterion(pred, labels[:, 1:], masks[:, 1:])
            loss.backward()
            clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            if float(torch.__version__[:3]) > 0.5:
                infos['TrainLoss'] = loss.item()
            else:
                infos['TrainLoss'] = loss.data[0]
            if infos['iter'] % opt.print_log_interval == 0:
                elapsed_time = time.time() - t_start
                log_info = [('Epoch', infos['epoch']), ('Iter', infos['iter']), ('Loss', infos['TrainLoss'])]
                if opt.use_ss == 1:
                    log_info += [('ss_prob', opt.ss_prob)]
                log_info += [('Time', elapsed_time)]
                logger.info('%s', '\t'.join(
                    ['{}: {}'.format(k, v) for (k, v) in log_info]))
            infos['iter'] += 1

        infos['epoch'] = epoch + 1
        checkpoint_checked = False
        learning_rate = optimizer.param_groups[0]['lr']
        logger.info('===> Learning rate: %f: ', learning_rate)
        # 测试时仅使用主进程
        if not ddp or (ddp and dist.get_rank() == 0):
            if (infos['epoch'] >= opt.save_checkpoint_from and infos[
                'epoch'] % opt.save_checkpoint_every == 0 and not checkpoint_checked):
                # evaluate the validation performance
                results = validate(model, criterion, val_loader, opt, mode='val')
                logger.info('Validation output: %s', json.dumps(results['scores'], indent=4, sort_keys=True))
                infos.update(results['scores'])

                check_model(model, opt, infos, infos_history)
                checkpoint_checked = True

        lr_scheduler.step()
    print('Training Finish')
    return infos


def indices_to_sentences(indices_list, ix_to_word):
    sentences = []  # 初始化句子列表
    for sentence_array in indices_list:
        for index_array in sentence_array:
            # 确保index_array是一个列表
            index_array = index_array.tolist() if isinstance(index_array, np.ndarray) else index_array

            # 去除索引为1和0的部分
            filtered_indices = [index for index in index_array if index not in [0, 1]]

            # 将剩余的索引转换为词汇表中的单词
            words = [ix_to_word.get(index, ix_to_word.get(2, '<unk>')) for index in filtered_indices]

            # 将单词列表组合成句子
            sentence = ' '.join(words)
            sentences.append(sentence)
            break  # 因为labels是双层列表，但似乎我们只关心每个双层列表的第一个元素
    return sentences


def language_eval(predictions, cocofmt_file, opt):
    logger.info('>>> Language evaluating ...')
    tmp_checkpoint_json = os.path.join(
        opt.model_file + str(uuid.uuid4()) + '.json')
    json.dump(predictions, open(tmp_checkpoint_json, 'w'))
    lang_stats = utils.language_eval(cocofmt_file, tmp_checkpoint_json)
    os.remove(tmp_checkpoint_json)
    return lang_stats


def validate(model, criterion, loader, opt, mode='val'):
    if mode=='val':
        batch_size=opt.test_batch_size
    else:
        batch_size=opt.test_batch_size
    model.eval()
    predictions = []
    for i, (feats, labels) in enumerate(loader):
        if torch.cuda.is_available():
            feats = feats.cuda()
            # motion = motion.cuda()
        with torch.no_grad():
            t_start = time.time()
            seq, logseq = model.module.sample(feats, {'beam_size': opt.beam_size})
            logger.info("Inference time: %f, batch_size: %d" % ((time.time() - t_start) / batch_size, batch_size))
            sents = utils.decode_sequence(opt.vocab, seq)

            for jj, sent in enumerate(sents):
                if mode == 'val':
                    entry = {'image_id': opt.train_num_videos + i * opt.batch_size + jj, 'caption': sent}
                else:
                    entry = {'image_id': opt.train_num_videos + opt.val_num_videos + i * opt.test_batch_size + jj, 'caption': sent}
                predictions.append(entry)
                logger.debug('[%d] video %s: %s' % (jj, entry['image_id'], entry['caption']))
    results = {}
    lang_stats = {}

    if opt.language_eval == 1:
        logger.info('>>> Language evaluating ...')
        tmp_checkpoint_json = os.path.join(opt.model_file + str(uuid.uuid4()) + '.json')
        json.dump(predictions, open(tmp_checkpoint_json, 'w'))
        lang_stats = utils.language_eval(getattr(opt, f'{mode}_cocofmt_file', None), tmp_checkpoint_json)
        # lang_stats = language_eval(sample_seqs=predictions, groundtruth_seqs=gts)
        os.remove(tmp_checkpoint_json)

    results['predictions'] = predictions
    results['scores'] = {}
    results['scores'].update(lang_stats)
    return results




def test(model, criterion, loader, opt):
    results = validate(model, criterion, loader, opt, mode='test')
    logger.info('Test output: %s', json.dumps(results['scores'], indent=4))

    json.dump(results, open(opt.result_file, 'w'))
    logger.info('Wrote output caption to: %s ', opt.result_file)

from collections import OrderedDict, defaultdict
def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    return {'BLEU': avg_bleu_score, 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}

if __name__ == '__main__':

    opt = opts.parse_opts()  # 参数设置
    if opt.local_rank >= 0:
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')
        ddp = True
    else:
        ddp = False

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()), format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))

    # Set the random seed manually for reproducibility.
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    train_opt = {'label_h5': opt.train_label_h5,
                 'batch_size': opt.batch_size,
                 'feat_h5': opt.train_feat_h5,
                 'cocofmt_file': opt.train_cocofmt_file,
                 'bcmrscores_pkl': opt.train_bcmrscores_pkl,
                 'eval_metric': opt.eval_metric,
                 'seq_per_img': opt.train_seq_per_img,
                 'num_chunks': opt.num_chunks,
                 'use_resnet_feature': opt.use_resnet_feature,
                 'use_c3d_feature': opt.use_c3d_feature,
                 'use_audio_feature': opt.use_audio_feature,
                 'use_global_local_feature': opt.use_global_local_feature,
                 'use_long_range': opt.use_long_range,
                 'use_short_range': opt.use_short_range,
                 'use_local': opt.use_local,
                 'mode': 'train'
                 }

    val_opt = {'label_h5': opt.val_label_h5,
               'batch_size': opt.test_batch_size,
               'feat_h5': opt.val_feat_h5,
               'cocofmt_file': opt.val_cocofmt_file,
               'seq_per_img': opt.test_seq_per_img,
               'num_chunks': opt.num_chunks,
               'use_resnet_feature': opt.use_resnet_feature,
               'use_c3d_feature': opt.use_c3d_feature,
               'use_audio_feature': opt.use_audio_feature,
               'use_global_local_feature': opt.use_global_local_feature,
               'use_long_range': opt.use_long_range,
               'use_short_range': opt.use_short_range,
               'use_local': opt.use_local,
               'mode': 'test'
               }

    test_opt = {'label_h5': opt.test_label_h5,
                'batch_size': opt.test_batch_size,
                'feat_h5': opt.test_feat_h5,
                'cocofmt_file': opt.test_cocofmt_file,
                'seq_per_img': opt.test_seq_per_img,
                'num_chunks': opt.num_chunks,
                'use_resnet_feature': opt.use_resnet_feature,
                'use_c3d_feature': opt.use_c3d_feature,
                'use_audio_feature': opt.use_audio_feature,
                'use_global_local_feature': opt.use_global_local_feature,
                'use_long_range': opt.use_long_range,
                'use_short_range': opt.use_short_range,
                'use_local': opt.use_local,
                'mode': 'test'
                }

    train_dataset = CaptionDataset(train_opt)
    val_dataset = CaptionDataset(val_opt)
    test_dataset = CaptionDataset(test_opt)

    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_caption)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=8, collate_fn= collate_fn_caption)

    # for i, (video_features,labels) in enumerate(val_dataloader):
    #     print(1)
    # train_loader = DataLoader(train_opt)
    # val_loader = DataLoader(val_opt)
    # test_loader = DataLoader(test_opt)
    #
    opt.vocab = train_dataset.get_vocab()
    opt.vocab_size = train_dataset.get_vocab_size()
    opt.seq_length = train_dataset.get_seq_length()
    opt.feat_dims = train_dataset.get_feat_dims()
    opt.history_file = opt.model_file.replace('.pth', '_history.json', 1)
    opt.train_num_videos = train_dataset.get_num_videos()
    opt.val_num_videos = val_dataset.get_num_videos()

    logger.info('Building model...')
    model = CaptionModel_clip(opt).cuda()
    #---------
    # for param in model.decoder.parameters():
    #     param.requires_grad = False
    #---------
    if ddp:
        model = DDP(model, device_ids=[opt.local_rank], find_unused_parameters=True)
        gpu_num = torch.distributed.get_world_size()
    else:
        gpu_num = 1
    opt.learning_rate = opt.learning_rate * gpu_num

    xe_criterion = CrossEntropyCriterion()
    rl_criterion = RewardCriterion()

    if torch.cuda.is_available():
        # model.cuda()
        xe_criterion.cuda()
        rl_criterion.cuda()

    logger.info('Start training...')
    start = datetime.now()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epochs, eta_min=0, last_epoch=-1)
    ## optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)

    infos = train(model, xe_criterion, optimizer,lr_scheduler, train_loader, val_loader, opt, rl_criterion=rl_criterion)
    logger.info(
        'Best val %s score: %f. Best iter: %d. Best epoch: %d',
        opt.eval_metric,
        infos['best_score'],
        infos['best_iter'],
        infos['best_epoch'])

    logger.info('Training time: %s', datetime.now() - start)

    if not ddp or (ddp and dist.get_rank() == 0):
        if opt.result_file:
            logger.info('Start testing...')
            start = datetime.now()

            logger.info('Loading model: %s', opt.model_file)
            checkpoint = torch.load(opt.model_file)
            model.load_state_dict(checkpoint['model'])

            test(model, xe_criterion, test_loader, opt)
            logger.info('Testing time: %s', datetime.now() - start)
