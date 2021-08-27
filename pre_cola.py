import pdb

import torchvision.transforms as transforms
from tran_cola import Resize,FormatShape,Normalize,ThreeCrop,ToTensor,Compose,RandomHorizontalFlip
from PIL import Image
from pathlib import Path
import argparse
import torch.utils.data as data
import os
import copy
import warnings
# from spatial_transforms import ()
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data:
        class_labels_map[class_label['label_name']] = class_label['label_id']
        index += 1
    return class_labels_map

def get_database(root_path,annotation_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in enumerate(annotation_path):
        value_split = value.replace('\n', ' ').split(' ')
        label_name = value_split[0].split('/')[0]
        video_id = value_split[0].split('/')[1]
        label_id = value_split[2]
        total_frame = value_split[1]

        video_ids.append(video_id)
        label = {
            'label_name':label_name,
            'label_id': label_id,
            'total_frame': total_frame
        }
        annotations.append(label)
        video_paths.append(video_path_formatter(root_path,value_split[0]))

    return video_ids, video_paths, annotations

def image_name_formatter(x):
    return f'img_{x:05d}.jpg'

class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, total_frames):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """


        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        return frame_inds

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []

        for shape in range(frame_indices.shape[0]):
            for i in frame_indices[shape]:
                image_path = video_path / self.image_name_formatter(i+1)
                # pdb.set_trace()
                if image_path.exists():
                    video.append(self.image_loader(image_path))

        return video

class VideoDatasetMultiClips(data.Dataset):

    def __init__(self,
                data_root_val,
                ann_file_val,
                 pre_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, video_id:
                                       root_path / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label',
                 bts=1):
        self.data, self.class_names = self.__make_dataset(
            data_root_val, ann_file_val, video_path_formatter)
        self.target_type = target_type
        self.pre_transform = pre_transform
        self.bts = bts

        if video_loader is None:
            pdb.set_trace()
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

    def __make_dataset(self, root_path, annotation_path,
                       video_path_formatter):
        video_ids, video_paths, annotations = get_database(
            root_path, annotation_path, video_path_formatter)
        class_to_idx = get_class_labels(annotations)

        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name
        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            # pdb.set_trace()
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))


            label = annotations[i]['label_name']
            label_id = class_to_idx[label]

            video_path = video_paths[i]
            if not video_path.exists():
                continue
            total_frame = int(annotations[i]['total_frame'])
            self.sample = SampleFrames(clip_len=8,frame_interval=8,num_clips=1,test_mode=True)
            frame_indices = self.sample(total_frame)

            sample = {
                'video': video_path,
                'video_id': video_ids[i],
                'frame_indices': frame_indices,
                'label': label_id
            }
            dataset.append(sample)
        return dataset, idx_to_class
    def __loading(self, path, video_frame_indices):
        segments = []
        clip = self.loader(path, video_frame_indices)
        # print('len(clip):',len(clip))
        if self.pre_transform is not None:
            # ttmmpp = []
            # for img in clip:
            #     mm = self.pre_transform(img)
            #     pdb.set_trace()
            clip = self.pre_transform(clip)
            #(3,3,256,256)

            # clip = self.pre_transform(clip)
        # 3,3,8,256,256

        num_clips = self.sample.num_clips
        clip_len = self.sample.clip_len
        imgs = clip.reshape((-1, num_clips, clip_len) + clip.shape[1:])  # 3,10,8,256,256
        # N_crops x N_clips x L x H x W x C
        imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4)) # 3,10,3,8,256,256img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]e
        # N_crops x N_clips x C x L x H x W
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # 30,3,8,256,256

        return imgs

    def __getitem__(self, index):
        clip_all = []
        targets = []
        for i in range(self.bts):
            num = index * self.bts + i
            print(num)
            # if num > len(self.data)-1:
            #     num = num % len(self.data)
            path = self.data[num]['video']
            # pdb.set_trace()
            video_frame_indices = self.data[num]['frame_indices']
            clips = self.__loading(path, video_frame_indices)

            clip_all.append(clips)
            targets.append(self.data[num]['label'])

        clip_all_t = torch.stack(clip_all)
        return clip_all_t, targets

def get_inference_data(data_root_val,
                       ann_file_val,
                       pre_transform=None,
                       bts=1):


    # from datasets.loader import ImageLoaderAccImage
    # loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
    loader = VideoLoader(image_name_formatter)
    video_path_formatter = (
        lambda root_path, video_id: root_path / video_id)

    inference_data = VideoDatasetMultiClips(
        data_root_val,
        ann_file_val,
        pre_transform = pre_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter,
        bts = bts
        )

    return inference_data, collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of 3D-ResNets')
    parser.add_argument('--data_root_val',
                    default='/home/qiong/code/huawei/mmaction2-master/data/ucf101/rawframes',
                    type=Path,
                    help='Directory path of videos')

    parser.add_argument('--ann_file_val',
                        default=f'/home/qiong/code/huawei/mmaction2-master/data/ucf101/ucf101_val_split_1_rawframes.txt',
                        type=Path,
                        help='Annotation file path')
    parser.add_argument(
        '--dataset_type',
        default='RawframeDataset',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')

    parser.add_argument('--output_path',
                default=f'/home/qiong/code/huawei/mmaction2-master/data/pre_base1_bs1_all/',
                type=Path,
                help='Directory path of binary output data')
    parser.add_argument(
        '--inference_batch_size',
        default=8,
        type=int,
        help='Batch Size for inference. 0 means this is the same as batch_size.')


    opt = parser.parse_args()
    inflie = open(opt.ann_file_val,'r',encoding= 'UTF-8')
    name = inflie.readlines()
    # kinetics nomalize
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    preprocess = [Resize(scale=(-1, 256))]
    preprocess.append(ThreeCrop(crop_size=256))
    preprocess.append(RandomHorizontalFlip(p=0))
    preprocess.append(Normalize(mean,std))
    # preprocess.append(FormatShape(input_format='NCTHW'))
    preprocess.append(ToTensor())

    preprocess = Compose(preprocess)

    inference_data, collate_fn = get_inference_data(
        opt.data_root_val,name, preprocess,bts = opt.inference_batch_size)


    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    inflie = open(opt.ann_file_val, 'r', encoding='UTF-8')
    name = inflie.readlines()

    for i, value in enumerate(inference_data):
        print(value[0].shape)
        video_ids = value[1]
        if len(value[1])==1:
            str_ids = str(video_ids[0])
        else:
            str_ids = '_'.join(str(i) for i in video_ids)
        batch_bin = value[0].cpu().numpy()
        print('preprocessing ' + str(video_ids))


        save_dir = str(opt.output_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + '/bin' + str(opt.inference_batch_size*i) + '-' + str(opt.inference_batch_size*(i+1)-1) + '_' + str_ids + '.bin'
        batch_bin.tofile(str(save_path))
        print( i, str(save_path), "save done!")

        print("-------------------next-----------------------------")

