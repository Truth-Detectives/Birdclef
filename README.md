# Final Submission CNN
*This notebook was used as a solution and submittable model to the birdCLEF2025 competition and achieved a score of 0.5.*

## Table of Contents
- Requirements
- Installation
- How to run notebook
- Directories
- Audio Parameters



## Requirements
*The package requirements for this notebook is as follows:*
[pandas](https://pypi.org/project/pandas/)
[numpy](https://pypi.org/project/numpy/)
[librosa](https://pypi.org/project/librosa/)
[sklearn](https://pypi.org/project/scikit-learn/)
[seaborn](https://pypi.org/project/seaborn/)
[torch](https://pypi.org/project/torch/)
[torchaudio](https://pypi.org/project/torchaudio/)


## Installation

### Install pandas

```bash
pip install pandas
```

### Install numpy

```bash
pip install numpy
```

### Install librosa

```bash
pip install librosa
```


### Install sklearn

```bash
pip install scikit-learn
```


## Install torch
```bash
pip install torch
```

## Install torchaudio

```bash
pip install torchaudio
```

## How to run notebook
*In order to run the segments of code. The user must click the necessary cell and then click the play button in the top navigation bar.*

## Directories
*These were used to retrieve audio .ogg files and relevant data such as row_ids.*

```python
test_directory = "/kaggle/input/birdclef-2025/test_soundscapes"
submission = "/kaggle/input/birdclef-2025/sample_submission.csv"
train_file = "/kaggle/input/birdclef-2025/train.csv"
taxonomy_file = "/kaggle/input/birdclef-2025/taxonomy.csv"

```

## Audio Parameters

```python
 sample_rate: int = 32000
    max_freq: int = 16000
    min_freq: int = 20
```
