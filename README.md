# 논문 서론 작성 AI

![python version badge](https://img.shields.io/badge/python-3.8.12-red)

## Team Members

- [김서린](https://github.com/Seorin-Kim) : 응용통계학과 </br>
- [서희재](https://github.com/linkyouhj) : 컴퓨터공학과 </br>
- [이재용](https://github.com/jaeyonggy) : 통계학과 </br>

<br>

## Introduction

저희는 서론에서 쓸 문장을 받으면 그에 맞는 문장들을 생성하는 AI를 만들었습니다. 논문은 주로 두괄식으로 쓰이기 때문에 첫번째 문장은 상당히 중요하므로 입력받는 텍스트는 사용자가 첫번째 문장만 잘 작성하면 됩니다. 또한 생성된 문장의 표절과 부자연스러움 등의 문제를 해결하기 위하여 paraphrasing을 이용하여 최종 문장들을 만들어냅니다.

문장 생성에서 쓰이게 될 모델은 KoGPT-2로 국내 논문 텍스트의 서론 부분만을 전이학습시켰습니다. 따라서 저희 모델은 서론 작성에 특화되었다고 보면 됩니다. Paraphrasing에서 쓰이게 될 모형은 이미 만들어진 모형에 한국어 paired 데이터셋을 전이학습시켰습니다.

<br>

## Installation

```console
git clone https://github.com/CUAI-CAU/2022_Summer_NLP_T12.git
pip install -r requirements.txt
```

requirements.txt 참고 바랍니다.

<br>

## Structure

![flow chart](https://user-images.githubusercontent.com/86909645/183421395-e5ae469c-ce6f-468d-af59-55fd75f57b6b.jpg)


- Text generation : [KoGPT2](https://github.com/SKT-AI/KoGPT2) 모델을 활용하여 한글 논문의 서론 텍스트를 학습시켰습니다.

- Paraphrasing : [한국어 Paraphrasing](https://github.com/L0Z1K/para-Kor) 모델을 활용하여 한국어 paraphrase paired 데이터셋을 학습시켰습니다.



<br>

## Text Generation Training

```console
$ python script/train.py --train_gen --train_gen_file /path/to/the/train/file --gen_epochs number
```
- Example:

```console
$ python script/train.py --train_gen --train_gen_file gen_example.txt --gen_epochs 2
```

## Paraphrasing Training

```console
$ python script/train.py --train_para --train_file /path/to/the/train/file
```
- Example:

```console
$ python script/train.py --gpus 1 --train_para --accelerator ddp --train_file para_example.csv
```
- 주의: 한국어 paraphrase paired 데이터셋은 A와 B라는 컬럼을 가져야 합니다. [para_example.csv](https://github.com/CUAI-CAU/2022_Summer_NLP_T12/blob/main/para_example.csv) 참고 바랍니다.

<br>

## Testing

문장을 input으로 받으면 그에 기반해서 여러 문장들을 생성한 후, 생성된 각 문장을 paraphrasing하여 최종 output을 제공합니다.

```console
$ python script/train.py --test --model_params_gen /path/to/the/gen_model/file --model_params_para /path/to/the/para_model/file --text_file /path/to/the/input/txt
```
- Example:

```console
$ python script/train.py --test --model_params_gen gen_finetune_2.pkl --model_params_para paraKor-epoch=50-train_loss=18.75.ckpt --text_file input_example.txt
```

<br>

## Demo

Demo.ipynb로 예시 결과를 보였습니다.

### Result

```console
Generated text with the given sentence:
인공지능이 글을 대신 써준다면 정말 편할 것입니다.
하지만 아직은 인공 지능의 연구가 활발하지 않은 이유는 무엇일까? 이 글에서는 인공 지능을 어떻게 정의하고 어떤 방식으로 구현해야 하는지에 대해 논의하고자 한다.
이를 위해 먼저 인공 신경망의 기본 개념과 특징에 대해서 살펴보고, 다음으로 인공신경망을 이용한 한국어 구문 분석 시스템과 그 결과를 살펴본다.
마지막으로 결론에서 본 논문의 끝을 맺는다 최근 들어 스마트폰, 태블릿 PC 등 모바일 기기의 사용이 보편화되면서 다양한 형태의 데이터가 생성되고 있다.
이러한 데이터를 수집, 가공하여 새로운 가치를 창출하는 빅데이터 시대가 도래하면서 대용량 데이터 처리에 대한 관심이 높아지고 있다[1].

Generated text after paraphrasing:
인공지능이 글을 대신 써준다면 정말 편할 것입니다.
하지만 아직 많은 연구들이 활발하게 이루어지지 않고 있다.
이를 위하여 먼저 한국어의 구문을 분석하고, 이를 바탕으로 한국어 문법 분석 시스템을 구축한 후 이를 토대로 한국어문법 분석을 실시하였다.
끝으로 결론을 맺고자 한다 최근 스마트폰이나 태블릿 등의 모바일기기의 사용량이 증가함에 따라 다양한 종류의 데이터 생성이 증가하고 있는 추세이다.
이러한 데이터 수집과 가공을 통한 새로운 가치 창출이 빅데이터 시대 도래에 따른 빅데이터 시대의 도래와 맞물려 빅데이터 시대를 맞이하게 되었다.
```

<br>

## Reference and related works

- https://github.com/SKT-AI/KoGPT2
- https://github.com/L0Z1K/para-Kor
- https://github.com/ttop32/KoGPT2novel

### 데이터 출처

- Text generation용: [AIDA - 국내 논문 전문 텍스트 데이터셋](https://aida.kisti.re.kr/data/19b111b4-03a5-40e4-87bd-844590a11202)

- Paraphrasing용:  [AI Hub - 한국어-영어 번역 말뭉치(사회과학)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=125), [AI Hub - 한국어-영어 번역 말뭉치(기술과학)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=124)

<br>




