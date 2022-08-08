# 논문 서론 작성 AI

## Team Members

- 김서린 : 응용통계학과 </br>
- 서희재 : 컴퓨터공학과 </br>
- 이재용 : 통계학과 </br>

<br>

## Installation

```console
git clone 주소
pip install -r requirements.txt
```

requirements.txt 참고 바랍니다.

<br>

## Structure

저희 모델은 두 구조로 나뉩니다: Text generation, Paraphrasing.

### Text generation

- [KoGPT2](https://github.com/SKT-AI/KoGPT2) 모델을 활용하여 한글 논문의 서론 텍스트를 학습시켰습니다.

### Paraphrasing

- [한국어 Paraphrasing](https://github.com/L0Z1K/para-Kor) 모델을 활용하여 한국어 paraphrase paired 데이터셋을 학습시켰습니다.

### Flow chart

Input 문장 -> (Text generation -> Paraphrasing) -> 서론의 한 문단 작성 (그림 추가할 예정)

<br>

## Training

### Text generation

```console
$ python script/train.py --train_gen --train_gen_file /path/to/the/train/file --gen_epochs number
```

- Example:

```console
$ python script/train.py --train_gen --train_gen_file gen_example.txt --gen_epochs 2
```

### Paraphrasing

```console
$ python script/train.py --train_para --train_file /path/to/the/train/file
```
- Example:

```console
$ python script/train.py --gpus 1 --train_para --accelerator ddp --train_file para_example.csv
```

- 주의

한국어 paraphrase paired 데이터셋은 A와 B라는 컬럼을 가져야 합니다. para_example.csv 참고 바랍니다.

<br>

## Testing

```console
$ python script/train.py --test --model_params_gen /path/to/the/gen_model/file --model_params_para /path/to/the/para_model/file --text_file /path/to/the/input/txt
```
- Example:

```console
$ python script/train.py --test --model_params_gen gen_finetune_2.pkl --model_params_para paraKor-epoch=50-train_loss=18.75.ckpt --text_file input_example.txt
```

<br>

## Demo

Demo.ipynb(추가할 예정)로 Google Colab에서 어떻게 돌리는지 참고 바랍니다.

### Result

(그림 추가할 예정)

<br>

## Reference and related works

- https://github.com/SKT-AI/KoGPT2
- https://github.com/L0Z1K/para-Kor
- https://github.com/ttop32/KoGPT2novel

### 데이터 출처

- Text generation용: [AIDA - 국내 논문 전문 텍스트 데이터셋](https://aida.kisti.re.kr/data/19b111b4-03a5-40e4-87bd-844590a11202)

- Paraphrasing용:  

<br>




