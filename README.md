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
$ python script/train.py --train_gen --train_gen_file intro.txt --gen_epochs 2
```

### Paraphrasing

```console
$ python script/train.py --train_para --train_file /path/to/the/train/file
```
- Example:

```console
$ python script/train.py --gpus 1 --train_para --accelerator ddp --train_file para_example.csv
```

- Note

The paired dataset should be divided into two columns A and B. Check out para_example.csv.

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

Check out Demo.ipynb(추가할 예정) to see how it's run on google colab.

### Result

(그림 추가할 예정)

<br>

## Reference and related works

- https://github.com/SKT-AI/KoGPT2
- https://github.com/L0Z1K/para-Kor

<br>




