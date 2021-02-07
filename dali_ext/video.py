from itertools import islice
from typing import List, Dict

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch import Tensor


class SinglePipeDaliIterator:
    def __init__(self, *args, **kwargs):
        self.iter = DALIGenericIterator(*args, **kwargs)

    def __next__(self) -> Dict[str, Tensor]:
        return next(self.iter)[0]


class VideoPipe(Pipeline):
    READER_NAME = 'Reader'
    RETURN = ['frames', 'file_indexes', 'frame_starts']

    def __init__(
            self,
            file_names: List[str],
            batch_size: int,
            num_workers: int,
            device_id: int,
            sequence_length: int,
            seed: int=-1,
            stride: int=1,
            read_ahead: bool=False,
            shuffle: bool=False,
    ):
        super().__init__(batch_size, num_workers, device_id, seed)
        self.input = ops.VideoReader(
            device='gpu',
            sequence_length=sequence_length,
            enable_frame_num=True,
            filenames=file_names,
            labels=[],
            shard_id=0,
            num_shards=1,
            random_shuffle=shuffle,
            read_ahead=read_ahead,
            stride=stride
        )

    def define_graph(self):
        """
        :return:
            - frames: shape=(batch_size, sequence_length, height, width, channel=3), uint8
            - file_ids: shape=(batch_size, 1), int32
            - num_frames: shape=(batch_size, 1), int32
        """
        frames, file_ids, num_frames = self.input(name=self.READER_NAME)
        return frames, file_ids, num_frames


class VideoLoader:
    def __init__(self, last_batch_policy: str, *pipeline_args, **pipeline_kwargs):
        super().__init__()
        self.pipe = VideoPipe(*pipeline_args, **pipeline_kwargs)
        self.pipe.build()
        self.last_batch_policy = last_batch_policy

    def __iter__(self):
        return SinglePipeDaliIterator(
            pipelines=[self.pipe],
            output_map=VideoPipe.RETURN,
            reader_name=VideoPipe.READER_NAME,
            last_batch_policy=self.last_batch_policy
        )
