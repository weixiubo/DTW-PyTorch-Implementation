# DTW-PyTorch-Implementation

Efficient PyTorch-based Dynamic Time Warping (DTW) and FastDTW Operators

---

## Introduction

This project provides efficient and flexible implementations of Dynamic Time Warping (DTW) and FastDTW operators using PyTorch. It supports GPU acceleration and batch processing, making it suitable for time series alignment, speech recognition, and other sequence analysis tasks. The code is well-structured and thoroughly documented, inspired by OpenDBA and cuDTW.

---

## Features
- Standard DTW and FastDTW algorithms
- GPU acceleration (auto-detects CUDA)
- Batch sequence processing
- Optional normalization and bandwidth constraint
- Well-commented and easy to extend

---

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   ```
2. Install dependencies:
   ```bash
   pip install torch numpy
   ```

---

## Usage Example

```python
from DTW_PyTorch_Implementation import DTW, FastDTW
import torch

# Create two random sequences
seq1 = torch.randn(50, 5)
seq2 = torch.randn(40, 5)

# Initialize DTW
dtw = DTW(use_cuda=True, normalize=True)
distance = dtw(seq1, seq2)
print('DTW distance:', distance.item())

# Initialize FastDTW
fast_dtw = FastDTW(radius=5, use_cuda=True, normalize=True)
fast_distance = fast_dtw(seq1, seq2)
print('FastDTW distance:', fast_distance.item())
```

For more usage, see the `example_usage()` function in `DTW_PyTorch_Implementation.py`.

---

## Reference & Acknowledgement
- [OpenDBA](https://github.com/zhw12/opendba)
- [cuDTW](https://github.com/kyoustat/cudtw)
- PyTorch official documentation

---

## License
MIT 
