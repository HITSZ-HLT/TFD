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

Download  [**multimodal-datasets**](https://drive.google.com/file/d/1Xxgp-D2idEcds023iPilyCXYY4kF9tm8/view?usp=drive_link) and save it in `./multimodal-datasets`.



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
  author={Geng Tu, Ran Jing, Bin Liang, Min Yang, Kam-Fai Wong, and Ruifeng Xu},
  booktitle={Proceedings of The 2023 Conference on Empirical Methods in Natural Language Processing, (EMNLP 2023)},
  year={2023}
}
```

## Credits
The code of this repository partly relies on [EmoBERTa](https://github.com/tae898/erc) and [CORSAIR](https://github.com/qianc62/Corsair). I would like to show my sincere gratitude to the authors behind these contributions.
