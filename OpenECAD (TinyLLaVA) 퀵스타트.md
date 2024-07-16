# TinyLLaVA 프레임워크를 이용한 OpenECAD 모델 구현 - 퀵스타트 가이드
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
Gemma는 HuggingFace(https://huggingface.co/google/gemma-2b-it)에서 무료로 사용할 수 있지만, 라이센스 인증 절차를 밟아야 합니다.

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
```bash
dataset/openecad$ huggingface-cli download --repo-type dataset --local-dir . Yuki-Kokomi/OpenECAD-Dataset images_2d_1.7z images_2d_2.7z
```
다운로드가 완료되면 `7z`로 압축 해제를 해줍니다 (`7z` 없다면 설치)
```bash
sudo apt install p7zip-full  # sudo 사용 주의!
dataset/openecad$ 7za x images_2d_1.7z  # 압축 해제 (참고: 오래걸림)
```

텍스트 데이터(CAD python code) 다운로드 (작업위치: `dataset/text_files`)
```bash
dataset/text_files$ huggingface-cli download --repo-type dataset --local-dir . Yuki-Kokomi/OpenECAD-Dataset data_2d_1.json data_2d_2.json
```


