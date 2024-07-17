# TinyLLaVA 프레임워크를 이용한 OpenECAD 모델 구현 - 퀵스타트 가이드
0. [TinyLLaVA 환경 구축 & Quick Inference](#0-tinyllava-환경-구축--quick-inference)

    0-1. [환경 구축](#0-1-환경-구축)

    0-2. [HuggingFace를 이용해 사전학습된 모델 불러오기 & Visual Question-Answering 추론 수행](#0-2-huggingface를-이용해-사전학습된-모델-불러오기--visual-question-answering-추론-수행)

    0-3. [(추가). Gemma 인증받기](#0-3-추가-gemma-인증받기)

1. [사전학습된 OpenECAD 모델 불러오기 & 추론 수행](#1-사전학습된-openecad-모델-불러오기--추론-수행)
   
    1-1. [OpenECAD 데이터셋 다운로드 (HuggingFace)](#1-1-openecad-데이터셋-다운로드-huggingface)

    1-2. [사전학습된 OpenECAD 모델 & 전처리기 불러오기](#1-2-사전학습된-openecad-모델--전처리기-불러오기)
---

## 0. TinyLLaVA 환경 구축 & Quick Inference
### 0-1. 환경 구축
참조: https://tinyllava-factory.readthedocs.io/en/latest/Installation.html

Github repository 설치
```bash
git clone https://github.com/TinyLLaVA/TinyLLaVA_Factory

cd TinyLLaVA_Factory
```

패키지 의존성 설치
```bash
conda create -n tinyllava_factory python=3.10 -y

conda activate tinyllava_factory

pip install --upgrade pip  # enable PEP 660 support

pip install -e .
```

추가 패키지 설치 (FlashAttention)
```bash
pip install flash-attn --no-build-isolation
```

### 0-2. HuggingFace를 이용해 사전학습된 모델 불러오기 & Visual Question-Answering 추론 수행
참조: https://github.com/TinyLLaVA/TinyLLaVA_Factory?tab=readme-ov-file#launch-demo-locally

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 사전학습 모델 불러오기
hf_path = 'tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()

# 모델 configuration & tokenizer 세팅
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)

# VQA 텍스트 프롬프트
prompt="What are these?"
# VQA 이미지 쿼리
image_url="http://images.cocodataset.org/val2017/000000039769.jpg"
# VQA 추론 수행
output_text, genertaion_time = model.chat(prompt=prompt, image=image_url, tokenizer=tokenizer)

print('model output:', output_text)
print('runing time:', genertaion_time)
```

위 Python 코드를 실행합니다.

만약 다음과 같은 에러가 발생한다면, 
> RuntimeError: Could not infer dtype of numpy.float32`
>
>During handling of the above exception, another exception occurred
>
>ValueError: Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.


numpy 버전을 확인해봅니다. 만약 numpy 버전이 2.0.0 이상이라면, numpy 버전으로 인해 발생할 가능성이 있으니, 다음과 같이 버전을 다운그레이드 해줍니다.

```bash
# numpy 버전 확인
pip list | grep numpy
```

```bash
# numpy 버전 1.23.5로 다운그레이드
pip install -U numpy==1.23.5
```


### 0-3 (추가). Gemma 인증받기
Gemma는 (HuggingFace)[https://huggingface.co/google/gemma-2b-it]에서 무료로 사용할 수 있지만, 라이센스 인증 절차를 밟아야 합니다.

다음 링크에 접속해 HuggingFace에 로그인(혹은 가입)한 뒤 Model card에 대한 Access를 얻습니다.

---

## 1. 사전학습된 OpenECAD 모델 불러오기 & 추론 수행

### 1-1. OpenECAD 데이터셋 다운로드 (HuggingFace)
먼저 `datasets` 패키지를 설치해줍니다.
> 만약 이 과정에서 numpy version이 업그레이드 됐다면, 다시 다운그레이드 해줍시다.

```bash
pip install datasets
```

`dataset` 디렉토리에 OpenECAD 데이터를 다운로드 받은 뒤 저장할 디렉토리 `openecad`를 만들어줍니다.
```bash
cd dataset
mkdir openecad && cd openecad
```

이미지 데이터(CAD 2D view) 다운로드 (작업위치: `dataset/openecad`)
- `images_2d_1.7z`와 `images_2d_2.7z`는 각각 2D sketch 이미지입니다.
- `images_3d.7z`는 3D 모델의 2d-view 이미지입니다.
```bash
dataset/openecad$ huggingface-cli download --repo-type dataset --local-dir . Yuki-Kokomi/OpenECAD-Dataset images_2d_1.7z images_2d_2.7z images_3d.7z
```
> :bulb: `$huggingface-cli download`로 다운로드받을 경우 `--local-dir`로 지정해준 디렉토리에 `.huggingface`라는 이름의 디렉토리가 새로 생겨있을 것입니다. 이 디렉토리는 추후에 데이터셋 업데이트가 있을 때 모든 데이터를 처음부터 다운로드받지 않아도 되게끔 해주는 디렉토리이므로, 가급적 삭제하지 않는 것을 추천합니다. (참고: https://huggingface.co/docs/huggingface_hub/guides/download)

다운로드가 완료되면 `7z`로 다운로드 받은 파일의 압축 해제를 해줍니다(`7z` 이미 깔려 있다면 skip).
```bash
sudo apt install p7zip-full  # 7z 설치. sudo 사용 주의!
dataset/openecad$ 7za x images_3d.7z  # 압축 해제 (참고: 오래걸림)
```
`images_2d_1.7z`, `images_2d_2.7z` 파일도 같은 명령어로 압축해제 할 수 있으나, 서로 파일명이 충돌할 수 있으니 별도의 분리된 디렉토리에 압축해제하는 것을 추천합니다.

텍스트 데이터(CAD python code) 다운로드 (작업위치: `dataset/text_files`)
- `data_2d_1.json`과 `data_2d_2.json`은 2D sketch 이미지에 대한 코드 데이터입니다.
- `data_3d.json`은 3D 모델에 대한 코드 데이터입니다.
- `data_3d_lite.json`은 3D 모델에 대한 코드 데이터로, 잘못 생성됐거나 context length를 초과하는 데이터를 제거한 버전인듯 합니다. 논문에서 학습 데이터로 사용한 데이터로 보입니다(총 130,239개).
- `data_3d_extend.json`은 3D 모델 코드 데이터에 코드 주석이 포함된 데이터입니다(총 75개).
```bash
dataset/text_files$ huggingface-cli download --repo-type dataset --local-dir . Yuki-Kokomi/OpenECAD-Dataset data_2d_1.json data_2d_2.json data_3d.json data_3d_extend.json data_3d_lite.json
```
> :bulb: 저자가 계속해서 데이터를 업데이트하고 있는 것으로 보입니다. [저자 프로필](https://huggingface.co/Yuki-Kokomi)을 종종 확인해 새로운 업데이트가 있는지 확인해줍시다.
>
> 새로운 데이터셋 업데이트가 있으면, 위의 `$huggingface-cli download` 명령어를 활용해 다운로드받아 줍니다.


### 1-2. 사전학습된 OpenECAD 모델 & 전처리기 불러오기
> [GitHub](https://github.com/cm8908/TinyLLaVA_CAD)에 올려둔 `test-openecad-2.4b.ipynb` 파일을 참고하세요.

모델, 전처리 모듈 불러오기
```python
from PIL import Image
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.data.image_preprocess import ImagePreprocess
from tinyllava.utils.constants import *
from tinyllava.utils.message import Message

# HuggingFace model path (https://huggingface.co/Yuki-Kokomi)
hf_path = 'Yuki-Kokomi/OpenECAD-SigLIP-2.4B'
# 모델, 토크나이저, 이미지처리, context length 불러오기
model, tokenizer, image_processor, context_len = load_pretrained_model(hf_path)
model.cuda()

# 텍스트, 이미지 전처리기 초기화
text_processor = TextPreprocess(tokenizer, 'gemma')
image_processor = ImagePreprocess(image_processor, model.config)

```

추론을 위한 예제 텍스트, 이미지 데이터 불러오기
```python
# 텍스트 데이터 파일 경로
text_path = 'dataset/text_files/data_3d_lite.json'
text_data = json.load(open(text_path))

data_id = text_data[0]['id']  # 데이터 ID
image_file = text_data[0]['image']  # 이미지 파일명
query = text_data[0]['conversations'][0]['value']  # 프롬프트
answer = text_data[0]['conversations'][1]['value']  # 정답 레이블

msg = Message()  # 메시지 객체 생성 (언어모델에 입력하기 위한 대화형 template을 만들어주는 역할)
msg.add_message(query)

result = text_processor(msg.messages, mode='eval')   # 토큰화
input_ids = result['input_ids'].unsqueeze(0).cuda()

# 이미지 파일 경로
image_path = f'dataset/openecad/images_3d/{image_file}'
image = Image.open(image_path).convert('RGB')
display(image)  # Jupyter notebook에서 이미지 출력

image_tensor = image_processor(image)  # 이미지 전처리(텐서화)
image_tensors = image_tensor.unsqueeze(0).half().cuda()
image_sizes = [image.size]
```

코드 생성 & 출력
```python
temperature = 0.2   # temperature가 높으면 더 다양한 문장이 생성됨, 낮으면 더 일관된 문장이 생성됨
output_ids = model.generate(inputs=input_ids,  # input token
                            images=image_tensors,  # image tensor
                            image_sizes=image_sizes,   # image size
                            do_sample=True if temperature > 0 else False,  # 확률적 sampling 여부
                            temperature=temperature,  # temperature
                            max_new_tokens=context_len,  # context length
                            use_cache=True
)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # 출력된 토큰을 텍스트로 디코딩

print(outputs[0])  # 터미널에 출력
print(outputs[0], file=open('output_example.md', 'w'))  # `output_example.md`라는 마크다운 파일에 출력
```
