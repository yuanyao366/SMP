import os
import torch.utils.data as data
import cv2
import sys

sys.path.append('..')
import random
import skvideo.io
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import argparse
import collections
from datasets import patch_region

envs = os.environ


class PredictDataset(data.Dataset):
    def __init__(self, root, mode="train", dataset="UCF-101", clip_len=16, video_transforms=None, image_transforms=None,
                 args=None):

        self.root = root
        self.mode = mode
        self.args = args
        self.dataset = dataset
        self.clip_len = clip_len
        self.video_transforms = video_transforms
        self.image_transforms = image_transforms
        if self.image_transforms:
            self.toPIL = transforms.ToPILImage()

        if self.dataset == 'UCF-101':
            self.split = '1'
            train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
            test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
            if mode == 'train':
                self.list = self.train_split
            else:
                self.list = self.test_split
        elif self.dataset == 'kinetics-400':
            self.list = []
            root_path = os.path.join(root, self.mode)
            classes = os.listdir(root_path)
            for class_ in classes:
                class_path = os.path.join(root_path, class_)
                class_files = os.listdir(class_path)
                for file in class_files:
                    videoname = os.path.join(class_path, file)
                    if ("webm" in videoname) == False:
                        self.list.append(videoname)
        elif self.dataset == 'kinetics-700':
            self.list = pd.read_csv(os.path.join(root, '%s_split.csv' % mode), header=None)

        self.batch_size = 8
        self.sample_step_list = [1, 2, 4, 8]
        self.recon_rate_list = [2]
        self.sample_retrieval = collections.OrderedDict()
        for i in range(len(self.recon_rate_list)):
            recon_rate = self.recon_rate_list[i]
            self.sample_retrieval[recon_rate] = self.sample_step_list[i:]
        self.recon_rate_label_batch = []
        recon_rate_label = np.random.randint(low=0, high=len(self.recon_rate_list))
        for i in range(self.batch_size):
            self.recon_rate_label_batch.append(recon_rate_label)
        self.MA_mode = args.ma_mode
        print('motion attention:{}'.format(self.MA_mode))
        print("dataset totalnum:", len(self.list))

    def __getitem__(self, index):

        if len(self.recon_rate_label_batch) == 0:
            recon_rate_label = np.random.randint(low=0, high=len(self.recon_rate_list))
            for i in range(self.batch_size):
                self.recon_rate_label_batch.append(recon_rate_label)
        recon_rate_label = self.recon_rate_label_batch.pop(0)
        recon_rate = self.recon_rate_list[recon_rate_label]

        videodata, sample_step_label = self.loadcvvideo_Finsert(index, recon_rate=recon_rate, sample_step=None)
        if self.image_transforms:
            target_clip = self.crop(np.array(videodata))
        else:
            target_clip = self.video_transforms(videodata)
            target_clip = torch.stack(target_clip).permute(1, 0, 2, 3)
        sample_step = self.sample_step_list[sample_step_label]
        sample_inds = torch.arange(0, len(videodata), step=sample_step)
        sample_clip = target_clip[:, sample_inds, :, :]

        if sample_step > 1:
            recon_step = int(sample_step / recon_rate)
            recon_inds = torch.arange(0, len(videodata), step=recon_step)
            recon_clip = target_clip[:, recon_inds, :, :]
            recon_flags = torch.ones(recon_clip.size(1), dtype=torch.int64)
        else:
            recon_step = sample_step
            recon_inds = torch.arange(0, len(videodata), step=recon_step)
            recon_clip1 = target_clip[:, recon_inds, :, :]
            recon_clip2 = target_clip[:, recon_inds, :, :]
            for i in range(len(recon_inds) - 1):
                recon_clip2[:, i, :, :] = (recon_clip1[:, i, :, :] + recon_clip1[:, i + 1, :, :]) / 2
            c, t, h, w = recon_clip1.size()
            recon_clip = torch.cat((recon_clip1.unsqueeze(dim=2), recon_clip2.unsqueeze(dim=2)), dim=2).reshape(c,
                                                                                                                2 * t,
                                                                                                                h, w)
            recon_flags = torch.zeros(recon_clip.size(1), dtype=torch.int64)
            up_rate = int(recon_rate / sample_step)
            recon_flags[[up_rate * k for k in range(t)]] = 1

        recon_clip_mask_len = int(recon_clip.size(1) / sample_step)
        recon_clip_mask = recon_clip[:, :recon_clip_mask_len, :, :]
        motion_mask_list = []
        for i in range(sample_step):
            clip16 = target_clip[:, i * 16:(i + 1) * 16, :, :]
            if self.MA_mode == 'DPAU':
                motion_mask = patch_region.getPatchLossMask_DPAU(clip16, recon_clip_mask, low=0.8, high=2.0)
            elif self.MA_mode == 'Gauss':
                motion_mask = patch_region.getPatchLossMask_Gauss(clip16, recon_clip_mask, low=0.8, high=2.0)
            elif self.MA_mode == 'OneMask':
                motion_mask = torch.ones_like(recon_clip_mask)
            motion_mask_list.append(motion_mask)
        motion_mask = torch.cat(motion_mask_list, dim=1)

        return sample_clip, recon_clip, sample_step_label, recon_rate, motion_mask, recon_flags

    def loadcvvideo_Finsert(self, index, recon_rate=None, sample_step=None):
        need = self.clip_len
        if self.dataset == 'UCF-101':
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)
        elif self.dataset == 'kinetics-400':
            fname = self.list[index]
        elif self.dataset == 'kinetics-700':
            v_class, v_name, frame_count = self.list.iloc[index]
            fname = os.path.join(self.root, self.mode, v_class, v_name)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if sample_step is None:
            if recon_rate:
                sample_step_proposal = self.sample_retrieval[recon_rate]
                proposal_idx = np.random.randint(low=0, high=len(sample_step_proposal))
                sample_step = sample_step_proposal[proposal_idx]
                sample_step_label = self.sample_step_list.index(sample_step)
            else:
                sample_step_label = np.random.randint(low=0, high=len(self.sample_step_list))
                sample_step = self.sample_step_list[sample_step_label]
        else:
            sample_step_label = self.sample_step_list.index(sample_step)

        sample_len = need * sample_step
        shortest_len = sample_len + 1
        while frame_count < shortest_len:
            index = np.random.randint(self.__len__())
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)
            capture = cv2.VideoCapture(fname)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        start = np.random.randint(0, frame_count - shortest_len + 1)
        if start > 0:
            start = start - 1
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while (sample_count < sample_len and retaining):
            retaining, frame = capture.read()
            if retaining is False:
                count += 1
                break
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1
        capture.release()

        while len(buffer) < sample_len:
            index = np.random.randint(self.__len__());
            print('retaining:{} buffer_len:{} sample_len:{}'.format(retaining, len(buffer), sample_len))
            buffer, sample_step_label = self.loadcvvideo_Finsert(index, recon_rate, sample_step)
            print('reload')

        return buffer, sample_step_label

    def crop(self, frames):
        print('aa')

        if self.video_transforms:
            frames = self.video_transforms(frames)

        if self.image_transforms:
            video_clips = []
            seed = random.random()
            for frame in frames:
                random.seed(seed)
                frame = self.toPIL(frame)
                frame = self.image_transforms(frame)
                video_clips.append(frame)
            clip = torch.stack(video_clips).permute(1, 0, 2, 3)
        else:
            clip = torch.tensor(frames)

        return clip

    def __len__(self):

        return len(self.list)




class ClassifyDataSet(data.Dataset):
    def __init__(self, root, mode="train", split="1", dataset="UCF-101", video_transforms=None, image_transforms=None):

        self.root = root
        self.mode = mode
        self.videos = []
        self.labels = []
        self.video_transforms = video_transforms
        self.image_transforms = image_transforms
        if self.image_transforms:
            self.toPIL = transforms.ToPILImage()
        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()])
        self.split = split
        self.dataset = dataset

        class_idx_path = os.path.join(root, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.mode == 'train':
            train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
            self.list = self.train_split
        else:
            test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
            self.list = self.test_split
        print('Dataset '+ self.dataset)
        print('Use split' + self.split)

    def loadcvvideo(self, fname, count_need=16):
        fname = os.path.join(self.root, 'video', fname)
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.dataset == 'HMDB-51':
            frame_count = frame_count - 1
        if count_need == 0:
            count_need = frame_count
        if count_need == 16:
            while(frame_count<count_need):
                capture.release()
                index = np.random.randint(self.__len__())
                fname = self.list[index]
                fname = os.path.join(self.root, 'video', fname)
                capture = cv2.VideoCapture(fname)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, frame_count - count_need + 1)
        if start > 0 and self.dataset == 'HMDB-51':
            start = start - 1
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while (sample_count < count_need and retaining):
            retaining, frame = capture.read()

            if retaining is False:
                count += 1

                break
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (171, 128))
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1

        capture.release()

        return buffer, retaining

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_split) - 1
        else:
            return len(self.test_split) - 1

    def __getitem__(self, index):
        if self.mode == 'train':
            videoname = self.train_split[index]
        else:
            videoname = self.test_split[index]

        if self.mode == 'train':
            videodata, retrain = self.loadcvvideo(videoname, count_need=16)
            while retrain == False or len(videodata) < 16:
                print('reload')
                index = np.random.randint(self.__len__())

                videoname = self.train_split[index]
                videodata, retrain = self.loadcvvideo(videoname, count_need=16)

            #videodata = self.randomflip(videodata)
            if self.image_transforms:
                video_clips = []
                if self.video_transforms:
                    videodata = self.video_transforms(np.array(videodata))
                seed = random.random()
                for frame in videodata:
                    random.seed(seed)
                    frame = self.toPIL(frame)
                    frame = self.image_transforms(frame)
                    video_clips.append(frame)
                clip = torch.stack(video_clips).permute(1, 0, 2, 3)
            else:
                video_clips = self.video_transforms(np.array(videodata))
                clip = torch.stack(video_clips).permute(1, 0, 2, 3)

        elif self.mode == 'test':
            fname = os.path.join(self.root, 'video', videoname)
            videodata = skvideo.io.vread(fname)
            clip = self.gettest(videodata)
        label = self.class_label2idx[videoname[:videoname.find('/')]]

        return clip, label - 1

    def randomflip(self, buffer):
        print('flip')
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def gettest(self, videodata):
        length = len(videodata)

        all_clips = []

        for i in np.linspace(8, length - 8, 10):
            clip_start = int(i - 8)
            clip = videodata[clip_start: clip_start + 16]
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms(frame)  # tensor [C x H x W]
                trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])

            all_clips.append(clip)
        return torch.stack(all_clips)











