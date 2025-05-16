# MLPClasiifier
*This notebook was used as a possible solution and submittable model to the birdCLEF2025 competition.*

## Table of Contents
- Requirements
- Installation
- How to run notebook
- Directories
- Dataframes
- Audio Feature Extraction
- Making the DataFrame for MLPClassifier

## Requirements
*The package requirements for this notebook is as follows:*
[pandas](https://pypi.org/project/pandas/)
[numpy](https://pypi.org/project/numpy/)
[matplotlib](https://pypi.org/project/matplotlib/)
[librosa](https://pypi.org/project/librosa/)
[noisereduced](https://pypi.org/project/noisereduce/)
[sklearn](https://pypi.org/project/scikit-learn/)
[seaborn](https://pypi.org/project/seaborn/)
[torch](https://pypi.org/project/torch/)
[scipy](https://pypi.org/project/scipy/)

## Installation

### Install pandas

```bash
pip install pandas
```

### Install numpy

```bash
pip install numpy
```
### Install matplotlib

```bash
pip install matplotlib
```

### Install librosa

```bash
pip install librosa
```

### Install noisereduced

```bash
pip install noisereduce
```

### Install sklearn

```bash
pip install scikit-learn
```

## Install seaborn

```bash
pip install seaborn
```

## Install torch
```bash
pip install torch
```

## Install scipy

```bash
pip install scipy
```

## How to run notebook
*In order to run the segments of code. The user must click the necessary cell and then click the play button in the top navigation bar.*

## Directories
*These were used to retrieve audio .ogg files.*

```python
train_data = "birdclef-2025/train_audio/"
test_data = "test_soundscapes/"
```

## DataFrames
*These were used to load relevant data such as filenames and labels.*

```python
names = pd.read_csv("new_train.csv")
tax = pd.read_csv("birdclef-2025/taxonomy.csv")
ids = pd.read_csv("birdclef-2025/sample_submission.csv")
```

## Audio Feature Extraction

```python
def extract_audio_features(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return {
        "rms": np.mean(librosa.feature.rms(y=y)).squeeze(),  
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)).squeeze(),  
        "flatness": np.mean(librosa.feature.spectral_flatness(y=y)).squeeze(),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)).squeeze(),
        "roll_off_high" : np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)).squeeze(),
        "roll_off_low" : np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)).squeeze(),
        "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))  
    }
```

## Making the DataFrame for MLPClassifier

```python
i=0
for path, result in results.items():
    audio_feat = result['features']
    df.loc[i,['path']] = path
    for feature in audio_feat.items():
        df.loc[i,['rms']] = audio_feat['rms']
        df.loc[i,['zcr']] = audio_feat['zcr']
        df.loc[i,['flatness']] = audio_feat['flatness']
        df.loc[i,['spectral_centroid']] = audio_feat['spectral_centroid']
        df.loc[i,['roll_off_high']] = audio_feat['roll_off_high']
        df.loc[i,['roll_off_low']] = audio_feat['roll_off_low']
        df.loc[i,['mfcc']] = audio_feat['mfcc']
    i=i+1
```
