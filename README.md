# DALI_extension
Useful code for NVIDIA DALI

## Video Pipeline

```python
from dali_ext import VideoLoader

video_files = [
    'XXX.mp4',
    'YYY.mp4',
    # ...
]

loader = VideoLoader(
    file_names=video_files,
    batch_size=1,
    num_workers=2,
    device_id=1,
    sequence_length=8,
    last_batch_policy='FILL'
)

for batch in loader:
    frames, file_indexes, frame_starts = batch['frames'], batch['file_indexes'], batch['frame_starts']
    # frames.shape=(batch_size, sequence_length, H, W, C), type=torch.Tensor, dtype=uint8
    # file_indexes.shape=(batch_size, 1), type=torch.Tensor, dtype=int32, the index of video in video_files, for example, index of XXX.mp4 is 0
    # frame_starts.shape=(batch_size, 1), type=torch.Tensor, dtype=int32, the frame_index of the first frame in video clips
```
