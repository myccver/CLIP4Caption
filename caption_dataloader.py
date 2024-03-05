import torch
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import pickle
import logging
logger = logging.getLogger(__name__)

class CaptionDataset(Dataset):
    def __init__(self, opt):
        self.iterator = 0
        self.epoch = 0

        self.batch_size = opt.get('batch_size', 128)
        self.seq_per_img = opt.get('seq_per_img', 1)
        self.word_embedding_size = opt.get('word_embedding_size', 512)
        self.num_chunks = opt.get('num_chunks', 1)
        self.mode = opt.get('mode', 'train')
        self.cocofmt_file = opt.get('cocofmt_file', None)

        self.use_global_local_feature = opt.get('use_global_local_feature', 0)
        self.label_path = opt['label_h5']
        self.dataset_name = self.get_dataset_name()

        # open the hdf5 info file
        logger.info('DataLoader loading h5 file: %s', opt['label_h5'])
        self.label_h5 = h5py.File(opt['label_h5'], 'r')

        self.vocab = [i for i in self.label_h5['vocab']]
        self.videos = [i for i in self.label_h5['videos']]

        self.ix_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.num_videos = len(self.videos)
        self.index = range(self.num_videos)

        # load the json file which contains additional information about the dataset
        self.feat_h5_files = opt['feat_h5']
        logger.info('DataLoader loading h5 files: %s', self.feat_h5_files)
        if self.use_global_local_feature == 1:
            self.feat_h5_clip = h5py.File(self.feat_h5_files[3], 'r')
        self.feat_dims = [self.feat_h5_clip['feats'].shape[-1]]

        # load in the sequence data
        if 'labels' in self.label_h5.keys():
            self.seq_length = self.label_h5['labels'].shape[1]
            logger.info('max sequence length in data is: %d', self.seq_length)

            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.label_h5['label_start_ix']
            self.label_end_ix = self.label_h5['label_end_ix']
            assert (self.label_start_ix.shape[0] == self.label_end_ix.shape[0])
            self.has_label = True
        else:
            self.has_label = False

        with open('data/captions/{}_{}_captions.pkl'.format(self.dataset_name,self.mode), 'rb') as file:
            self.captions = pickle.load(file)
        with open('data/metadata/{}_updated_vocab_whole.pkl'.format(self.dataset_name), 'rb') as f:
            self.vocab = pickle.load(f)
        self.ix_to_word = {i: w for i, w in enumerate(self.vocab)}

        # self.motion_data = []
        # with h5py.File('data/c3d_feature/{}_C3D_{}.hdf5'.format(self.dataset_name.upper(),self.mode)) as file:
        #     for dataset_name in file:
        #         data = file[dataset_name][:]
        #         temp_num = data.shape[0]
        #         sample_num = 5
        #         indices = np.linspace(0, len(data) - 1, num=sample_num,dtype=int)
        #
        #         # 根据索引采样数据
        #         sampled_data = data[indices]
        #
        #         # 将采样后的数据添加到motion_data列表中
        #         self.motion_data.append(sampled_data)


    def __del__(self):
        if self.use_global_local_feature == 1:
            self.feat_h5_clip.close()

        self.label_h5.close()


    def __getitem__(self, idx):
        if self.mode == 'train':
            # 获取对应的视频特征
            video_idx = self.label_h5['label_to_video'][idx]
            video_features = torch.tensor(self.feat_h5_clip['feats'][video_idx])

            # 获取对应的标签
            # labels = torch.tensor(self.label_h5['labels'][idx])
            # masks = torch.zeros_like(labels)
            # last_non_zero = len(labels) - (labels == 0).sum() - 1
            # masks[:last_non_zero + 2] = 1

            # 重新获取标签
            words = self.captions[idx].split()
            words.insert(0, '<start>')
            if len(words) > 29:
                words = words[:29]
            # index_list = [self.vocab.index(word) if word in self.vocab else self.vocab.index('<unk>') for word in words]
            index_list = []
            for word in words:
                if word in self.vocab:
                    index_list.append(self.vocab.index(word))
                else:
                    unk_index = self.vocab.index('<unk>')
                    # print(f"Word '{word}' not found. Using '<unk>' at index {unk_index}.")
                    index_list.append(unk_index)

            # 填充0直到列表长度为30
            padding_length = 30 - len(index_list)
            index_list.extend([0] * padding_length)
            # 转化为tensor
            labels = torch.tensor(index_list)
            masks = torch.zeros_like(labels)

            # 根据labels的非0数量，设置masks的值为1
            non_zero_count = (labels != 0).sum().item()
            masks[:non_zero_count + 1] = 1
            # motion_feature = torch.tensor(self.motion_data[video_idx],dtype=torch.float32)
            return video_features, labels, masks
        else:
            video_features = torch.tensor(self.feat_h5_clip['feats'][idx])
            labels = self.label_h5['labels'][self.label_start_ix[idx]: self.label_end_ix[idx]]
            labels = [item for item in labels]
            # motion_feature = torch.tensor(self.motion_data[idx], dtype=torch.float32)
            return video_features,labels

    def __len__(self):
        if self.mode == 'train':
            return len(self.label_h5['labels'])
        else:
            return len(self.videos)
    def get_vocab(self):
        return self.ix_to_word

    def get_vocab_size(self):
        return len(self.vocab)

    def get_feat_dims(self):
        return self.feat_dims

    def get_seq_length(self):
        return self.seq_length
    def get_num_videos(self):
        return self.num_videos

    def get_dataset_name(self):
        path_string = self.label_path

        # 使用'/'拆分字符串并获取'msvd_train_sequencelabel.h5'
        filename = path_string.split('/')[-1]

        # 使用'_'拆分字符串并获取'msvd'
        dataset_name = filename.split('_')[0]
        return dataset_name


def collate_fn_caption(batch):
    video_features = [item[0] for item in batch]
    # motion_features = [item[1] for item in batch]
    labels = [item[1] for item in batch]


    # 堆叠video_features
    video_features = torch.stack(video_features, 0)
    # motion_features = torch.stack(motion_features, 0)

    return video_features , labels
