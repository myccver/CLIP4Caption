# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from tranformer import *
import math
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, logprobs, reward):
        # import pdb; pdb.set_trace()
        logprobs = to_contiguous(logprobs).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        # add one to the right to count for the <eos> token
        mask = to_contiguous(torch.cat(
            [mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        # import pdb; pdb.set_trace()
        output = - logprobs * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output



class CrossEntropyCriterion(nn.Module):

    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, pred, target, mask):
        # truncate to the same size
        target = target[:, :pred.size(1)]
        mask = mask[:, :pred.size(1)]

        pred = to_contiguous(pred).view(-1, pred.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = -pred.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


# 创建位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
# 可学习的位置编码
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=40):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class FeatPool(nn.Module):

    def __init__(self, feat_dims, out_size, dropout):
        super(FeatPool, self).__init__()

        module_list = []
        for dim in feat_dims:
            module = nn.Sequential(
                nn.Linear(dim, out_size),
                nn.ReLU(),
                nn.Dropout(dropout))
            module_list += [module]
        self.feat_list = nn.ModuleList(module_list)

        # self.embed = nn.Sequential(nn.Linear(sum(feat_dims), out_size), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, feats):
        """
        feats is a list, each element is a tensor that have size (N x C x F)
        at the moment assuming that C == 1
        """
        if feats[0].size(2) == 20:
            # feats[0] = torch.cat(feats[0].permute(2, 0, 1, 3), 2)
            feats[0] = torch.mean(feats[0], 2)

        out = torch.cat([m(feats[i].squeeze(1))
                         for i, m in enumerate(self.feat_list)], 1)
        # pdb.set_trace()
        # out = self.embed(torch.cat(feats, 2).squeeze(1))
        return out


class FeatExpander(nn.Module):

    def __init__(self, n=1):
        super(FeatExpander, self).__init__()
        self.n = n

    def forward(self, x):
        if self.n == 1:
            out = x
        else:
            out = Variable(x.data.new(self.n * x.size(0), x.size(1)))
            for i in range(x.size(0)):
                out[i * self.n:(i + 1) * self.n] = x[i].expand(self.n, x.size(1))
        return out

    def set_n(self, x):
        self.n = x


class RNNUnit(nn.Module):

    def __init__(self, opt):
        super(RNNUnit, self).__init__()
        self. rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        if opt.model_type == 'standard':
            self.input_size = opt.input_encoding_size
        elif opt.model_type in ['concat', 'manet']:
            self.input_size = opt.input_encoding_size + opt.video_encoding_size

        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, state):
        output, state = self.rnn(xt.unsqueeze(0), state)
        return output.squeeze(0), state



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 21)
        )

    def forward(self, feature_1, feature_2):
        # print(feature_2.size())
        # batch_size = feature_2.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(feature_1)
        weights = F.softmax(energy.squeeze(1))
        print(weights.size())
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = feature_2 * weights.unsqueeze(-1).unsqueeze(-1)
        return outputs, weights


class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        # attention = self.softmax(energy)  # BX (N) X (N)
        attention = F.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class CaptionModel_clip(nn.Module):
    """
    A baseline captioning model
    """

    def __init__(self, opt):
        super(CaptionModel_clip, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0

        self.use_resnet_feature = opt.use_resnet_feature
        self.use_c3d_feature = opt.use_c3d_feature
        self.use_global_local_feature = opt.use_global_local_feature

        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        # vocab_embedding
        self.vocab_embedding = opt.vocab_embedding
        self.init_weights()
        self.feat_pool = FeatPool(self.feat_dims, self.num_layers * self.rnn_size, self.drop_prob_lm)
        self.feat_expander = FeatExpander(self.seq_per_img)

        self.video_encoding_size = self.num_feats * self.num_layers * self.rnn_size
        opt.video_encoding_size = self.video_encoding_size
        # self.core = RNNUnit(opt)
        # decoder_layer = TransformerDecoderLayer(
        #     d_model=768,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation="relu",
        #     normalize_before=False
        # )
        # self.decoder = TransformerDecoder(
        #     decoder_layer=decoder_layer,
        #     num_layers=1,
        #     norm=None,
        #     return_intermediate=False
        # )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=1,
            norm=None
        )
        # self.pos_encoder = PositionalEncoding(768)
        self.pos_encoder = LearnablePositionalEncoding(768)

        # self.motion_fc = nn.Sequential(nn.Linear(2048,768),nn.ReLU(),nn.Dropout(0.5))
        self.clip_fc = nn.Sequential(nn.Linear(768,768),nn.ReLU(),nn.Dropout(0.5))

        #----lstm--
        # self.lstm = LSTMModel(input_size=512, hidden_size=512, num_layers=2, bidirectional=False)
        # self.ca = ChannelAttention1D(num_frames=60,frame_features=512,reduction=4,keep_frames=40)

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        # self.embed.weight.data.uniform_(-initrange, initrange)
        with open(self.vocab_embedding, 'rb') as file:
            loaded_tensor = pickle.load(file)
        self.embed.weight.data = loaded_tensor
        self.embed.weight.requires_grad = False
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
    # def init_weights(self):
    #     initrange = 0.1
    #     self.embed.weight.data.uniform_(-initrange, initrange)
    #     self.logit.bias.data.fill_(0)
    #     self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if self.rnn_type == 'lstm':
            return (
                Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_())



    def forward(self, feats, seq):

        # fc_feats = self.feat_pool(feats)
        # fc_feats = self.lstm(feats)
        # feats = feats[:, torch.linspace(0, 59, 20).long(), :]

        # fc_feats = torch.mean(feats, dim=1)
        # fc_feats = torch.max(feats, dim=1)[0]
        # fc_feats = self.feat_expander(fc_feats)
        # seclected_feats = self.ca(feats)
        # fc_feats = torch.mean(seclected_feats,dim=1)

        # batch_size = fc_feats.size(0)
        # motion = self.motion_fc(motion)
        feats = self.clip_fc(feats)
        # feats = torch.cat([feats,motion],dim=1)
        batch_size = feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        sample_seq = []
        sample_logprobs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = seq.size(1) - 1
        #---------
        tgt_key_padding_mask = (seq == 0)
        seq_length = seq.size(1)
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(seq.device)
        seq_emb = self.embed(seq)
        #------位置编码-----
        # seq_emb = seq_emb * math.sqrt(768)  # 缩放嵌入
        # seq_emb = self.pos_encoder(seq_emb.transpose(0, 1)).transpose(0, 1)  # 添加位置编码
        #-------------
        # 可学习位置编码
        seq_emb = self.pos_encoder(seq_emb)
        #
        # feats = torch.mean(feats,dim=1,keepdim=True)
        output = self.decoder(tgt=seq_emb.transpose(0,1), memory=feats.transpose(0,1),tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask).transpose(0, 1)


        output = F.log_softmax(self.logit(self.dropout(output)), dim=-1)
        # 创建一个掩码，其中值为True的位置代表该列全为0
        all_zero_columns = (seq == 0).all(dim=0)

        # 获取第一个全为0的列的索引
        first_zero_column_index = torch.argmax(all_zero_columns.int())
        output = output[:, :first_zero_column_index]

        return output, output[-1] , output[-2]



    def sample(self, feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        expand_feat = opt.get('expand_feat', 0)

        if beam_size > 1:
            return self.sample_beam(feats,opt)



    def sample_beam(self, feats, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """

        beam_size = opt.get('beam_size', 5)
        feats = self.clip_fc(feats)

        # motion = self.motion_fc(motion)
        # feats = torch.cat([feats, motion], dim=1)
        batch_size = feats.size(0)
        #-----------


        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            new_seq = torch.zeros((beam_size, self.seq_length), dtype=torch.long).cuda()
            state = self.init_hidden(beam_size)
            fc_feats_k = feats[k].expand(
                beam_size,feats[k].size()[0] ,self.video_encoding_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            for token_idx in range(start_i, end_i):
                if token_idx == -1:
                    xt = fc_feats_k
                elif token_idx == 0:  # input <bos>
                    it = feats.data.new(beam_size).long().fill_(self.bos_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob

                            if float(torch.__version__[:3]) > 0.5:
                                candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data.item()
                                                      , 'r': local_logprob.item()})
                            else:
                                candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data[
                                    0], 'r': local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[
                            :token_idx - 1,
                            vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at
                            # vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append(
                                {'seq': beam_seq[:, vix].clone(), 'logps': beam_seq_logprobs[:, vix].clone(),
                                 'p': beam_logprobs_sum[vix], 'ppl': ppl})

                    # encode as vectors
                    it = beam_seq[token_idx - 1]
                    xt = self.embed(Variable(it.cuda()))

                if token_idx >= 1:
                    state = new_state


                # output, state = self.core(torch.cat([xt, fc_feats_k], 1), state)
                # logprobs = F.log_softmax(self.logit(output),dim=1)
                # if token_idx==2:
                #     print(1)
                new_seq[:, token_idx] = it.cuda()
                new_xt = self.embed(new_seq)
                tgt_key_padding_mask = (new_seq == 0)
                #------位置编码-----
                # new_xt = new_xt * math.sqrt(768)  # 缩放嵌入
                # new_xt = self.pos_encoder(new_xt.transpose(0, 1)).transpose(0, 1)  # 添加位置编码
                #------------------------------
                # 可学习位置编码
                new_xt = self.pos_encoder(new_xt)
                #
                # fc_feats_k = torch.mean(fc_feats_k,dim=1,keepdim=True)
                output = self.decoder(tgt=new_xt.transpose(0, 1), memory=fc_feats_k.transpose(0, 1),tgt_key_padding_mask=tgt_key_padding_mask).transpose(0, 1)
                # output = output[:,token_idx,:]
                logprobs = F.log_softmax(self.logit(output), dim=-1)
                logprobs = logprobs[:, token_idx, :]

            # self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, bidirectional):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        # x 的形状为 (batch, 60, 512)
        output, (hn, cn) = self.lstm(x)
        return torch.mean(output, dim=1)

class ChannelAttention1D(nn.Module):
    def __init__(self, num_frames, frame_features, reduction=4, keep_frames=5):
        super(ChannelAttention1D, self).__init__()
        self.num_frames = num_frames
        self.frame_features = frame_features
        self.keep_frames = keep_frames
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(num_frames, num_frames // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_frames // reduction, num_frames, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.module(x)
        # 找出权重最大的 keep_frames 个帧的索引
        _, indices = torch.topk(weights.mean(dim=2), self.keep_frames, dim=1)
        # 根据索引选择对应的帧
        selected_frames = torch.gather(x, 1, indices.unsqueeze(2).expand(-1, -1, self.frame_features))
        return selected_frames