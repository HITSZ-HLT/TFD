# TFD

> The official implementation for the conference of EMNLP 2023 paper *A Training-Free Debiasing Framework with Counterfactual Reasoning for Conversational Emotion Detection*.

<img src="https://img.shields.io/badge/Venue-EMNLP--23-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirements
* Python 3.9
* PyTorch 1.10.1
* Transformers 4.33.1
* CUDA 10.2
* pysentiment 0.2

## Preparation

Download  [**multimodal-datasets**]([https://drive.google.com/file/d/1Xxgp-D2idEcds023iPilyCXYY4kF9tm8/view?usp=drive_link]) and save it in `./multimodal-satasets`.



## Training & Evaluation

```sh
python train-erc-text.py
```
use train-erc-text.yaml to change the parameters.

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{,
  title={A Training-Free Debiasing Framework with Counterfactual Reasoning for Conversational Emotion Detection},
  author={},
  journal={},
  year={2023}
}
```

## Credits
The code of this repository partly relies on [erc](https://github.com/tae898/erc) and I would like to show my sincere gratitude to authors of it.
