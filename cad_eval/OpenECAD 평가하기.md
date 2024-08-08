# 0. 파인튜닝된 모델 불러와서 데이터 생성하기
> :warning: 본 매뉴얼에서는 먼저 OpenECAD 데이터셋을 `train` / `test` 셋으로 나눈 뒤에 `train` 데이터셋에 대해 **사전학습을 모두 마친 뒤의 상황**을 가정하고 있습니다.
## TinyLLaVA 스크립트를 이용해 `test` 데이터에 대해 inference 수행

만약 [OpenECAD 퀵스타트 가이드](https://github.com/cm8908/TinyLLaVA_CAD/blob/main/OpenECAD%20(TinyLLaVA)%20%ED%80%B5%EC%8A%A4%ED%83%80%ED%8A%B8.md)에 따라 파인튜닝을 잘 마쳤다면, `outputs` 디렉토리에 파인튜닝된 모델의 체크포인트 등이 포함된 디렉토리가 만들어져 있을 것입니다. 가령 파인튜닝 시 설정한 `--run_name`이 `MyModel`이라고 하면, `outputs/MyModel`가 아래와 같이 생성되어 있을 것입니다.
```
outputs/
+-- MyModel/
|   +-- connector/
|   |   +-- pytorch_model.bin
|   +-- language_model/
|   |   +-- pytorch_model.bin
|   |   +-- config.json
|   +-- vision_tower/
|   |   ...
|   +-- runs/
|   |   ...
|   +-- adapter_config.json
|   +-- config.json
|   ...
```

그럼 이제 `test` 데이터에 대해 모델 inference를 수행하기 위해 스크립트 파일을 준비해 줍니다.
> :bulb: [`scripts/eval/openecad.sh`](https://github.com/cm8908/TinyLLaVA_CAD/blob/main/scripts/eval/openecad.sh)를 참고하세요.

```bash
MODEL_PATH="path/to/outputs/MyModel"  # run_name이 포함된 디렉토리 경로
MODEL_NAME="MyModel"  # run name
EVAL_IMG_DIR="path/to/dataset/openecad/images_3d_test"  # 이미지 테스트 데이터
EVAL_TEXT_PATH="path/to/dataset/text_files/data_3d_lite_test.jsonl"  # 텍스트 테스트 데이터
MAX_TOKENS=2048
```

스크립트를 실행하세요. 단, `test` 데이터셋의 크기가 커 시간이 매우 오래걸릴 수 있습니다. `test` 데이터셋의 크기를 조금 줄이고 나서 (약 5,000개~10,000개) 실행하는 것을 권장합니다.
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/openecad.sh
```

Inference가 성공적으로 수행되고 나면, `outputs/MyModel/generated` 디렉토리에 `merge.jsonl`가 새로 생겨나있을 겁니다. 그 파일이 생성된 데이터가 담긴 파일입니다.

## 생성된 `.jsonl` 파일 -> `.py` -> `.step` 파일로 변환
`cad_eval/json_to_py.py` 파일을 아래와 같이 실행하여 생성된 jsonl 파일에 있는 데이터들을 python code들로 변환합니다 (경로는 자신의 모델이 위치한 경로에 맞게 지정해주세요).

```bash
python cad_eval/jsonl_to_py.py --jsonl_path outputs/MyModel/generated/merge.jsonl --save_dir outputs/MyModel/generated/pyfiles
```

이후 evaluation을 수행하기 위해, `.py` 코드들을 `.step` 포맷으로 변환합니다 (`pythonocc_operator/py2step.py` 실행).

```bash
python pythonocc_operator/py2step.py --src outputs/MyModel/generated/pyfiles -o outputs/MyModel/generated/step_files
```

위 과정이 완료되면, 아래와 같이 파일들이 생성되어 있을 것입니다.
```
...
|	MyModel/
|   ...
|   +-- generated/
|   |   +-- pyfiles/
|   |   |   +-- xxx.py
|   |   |   ...
|   |   +-- step_files/
|   |   |   +-- xxx.step
|   |   |   ...
|   |   +-- merge.jsonl
|   |   ...
```
# 1. Valid / Unique 평가하기
간단하게 `cad_eval/evaluate_3metrics.py`를 적절한 경로 설정과 함께 실행해주면 됩니다.

```bash
python cad_eval/evaluate_3metrics.py --src outputs/MyModel/generated/step_files --parallel --log_dir outputs/MyModel/generated  # --parallel 필수
```

코드 실행 후 `outputs/MyModel/generated` 디렉토리에 `3metrics.txt` 파일이 새로 생성되어 있을 겁니다.

# 2. MMD / COV / JSD 평가하기
## 2-1. `.step` 파일 -> `.ply` (point cloud)로 변환하기
MMD, COV, JSD는 모두 point cloud 기반으로 평가하기 때문에, `.step` 파일을 모두 point cloud(`.ply`)로 변환해주는 전처리 작업이 필요합니다.

이를 위해 `cad_eval/collect_gen_pc.py`를 실행해줍니다. `--output`에 지정한 디렉토리 경로에 `.ply` 파일들이 생성됩니다.
```bash
python cad_eval/collect_gen_pc.py --src outputs/MyModel/generated/step_files --output outputs/MyModel/generated/pc_files
```

실행 이후 파일구조:
```
...
|	MyModel/
|   ...
|   +-- generated/
|   |   ...
|   |   +-- pc_files/
|   |   |   +-- xxx.ply
|   |   |   ...
|   |   ...
```

## 2-2. Evaluation 코드 실행
```bash
python cad_eval/evaluate_gen_torch.py --src outputs/MyModel/generated/pc_files -g 0,1 -o outputs/MyModel/generated
```
코드 실행 이후 `outputs/MyModel/generated` 디렉토리에 `gen_pc_eval.txt` 파일이 생성될겁니다.