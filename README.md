# LibriSpeech Same-Different Word Discrimination Task

A tool to evaluate framewise speech features for word discrimination with dynamic time warping. It follows the method from [Rapid evaluation of speech representations for spoken term discovery](https://www.isca-speech.org/archive/interspeech_2011/carlin11_interspeech.html).

The word features are extracted from the LibriSpeech dataset after aligning with [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/). The alignments used can be found [here](https://github.com/nicolvisser/librispeech-samediff/releases/tag/v0.1).

Words are selected with the following criteria:

- 5 or more characters
- 0.5 seconds or more duration
- at least 2 occurrences in the subset

Summary of the selected words:

| Subset     | Word Count | Word Pairs | SWSP\* | SWDP\* | DWSP\*  | DWDP\*     |
| ---------- | ---------- | ---------- | ------ | ------ | ------- | ---------- |
| dev-clean  | 4,167      | 8,679,861  | 1,300  | 6,044  | 231,469 | 8,441,048  |
| dev-other  | 3,261      | 5,315,430  | 2,071  | 4,416  | 191,897 | 5,117,046  |
| test-clean | 4,910      | 12,051,595 | 1,689  | 6,895  | 319,360 | 11,723,651 |
| test-other | 3,561      | 6,338,580  | 1,615  | 4,422  | 216,799 | 6,115,744  |
| dev        | 7,428      | 27,583,878 | 3,371  | 18,642 | 423,366 | 27,138,499 |
| test       | 8,471      | 35,874,685 | 3,304  | 20,455 | 536,159 | 35,314,767 |

<sup><sub>
*SWSP = Same Word Same sPeaker<br>
*SWDP = Same Word Different sPeaker<br>
*DWSP = Different Word Same sPeaker<br>
*DWDP = Different Word Different sPeaker
</sub></sup>

## Installation

Ensure you have a working Python environment with `dtaidistance` installed.

```bash
conda create -n libri-sd -c conda-forge dtaidistance
```

```bash
conda activate libri-sd
```

To check if the `dtaidistance` library is built with C support, run the following command:

```bash
python -c "from dtaidistance import dtw; print(dtw.try_import_c(verbose=True))"
```

If not, check [installation and troubleshooting](https://dtaidistance.readthedocs.io/en/latest/usage/installation.html).

Install the tool from the root directory

```bash
pip install .
```

## Usage

### Data preparation

Ensure you have a directory containing a numpy file for each utterance in the evaluation set. The numpy files should be named with the utterance ID, e.g. `1272-128104-0000.npy`. The tool will recursively search for a file in the directory.

Each numpy file should contain a 2D array of shape `(T, F)` where `T` is the number of frames and `F` is the number of features per frame. The features should be properly aligned with the audio such that `T*feature_rate ≈ audio_duration`.

### Running the tool

From a terminal with the environment activated, run

```bash
libri-sd
```

then follow the prompts.

Alternatively, specify the options:

```
Usage: libri-sd [OPTIONS]

Options:
  --subset [dev-clean|dev-other|test-clean|test-other|dev|test]
                                  The LibriSpeech subset to use  [required]
  --feature-dir DIRECTORY         Directory containing the features
                                  [required]
  --feature-rate FLOAT            How many features per second  [required]
  --log-dir DIRECTORY             Directory to save logs results to
                                  [required]
  --run-name TEXT                 Name of subdirectory to save logs results
                                  to. If not specified, will use current
                                  timestamp
  --help                          Show this message and exit.
```

## Example Results

We evaluate the soft HuBERT features from this [repo](https://github.com/bshall/hubert) and [paper](https://ieeexplore.ieee.org/abstract/document/9746484) on the dev-clean set.

```
Average Precision:
SW:     0.8776
SWSP:   0.9088
SWDP:   0.8710

Precision Recall Break-Even Point:
SW:     0.8136
SWSP:   0.8538
SWDP:   0.8066
```

![precision recall curve example](./hubert-bshall-soft-dev-clean.png)
