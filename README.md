# GLIGEN_code

https://github.com/gligen/GLIGEN  
(*위 링크에서 GLIGEN 환경 세팅)

## Preprocess
1. `pip install -r requirements.txt` 실행
2. `labelme2caption.py` 파일을 `GLIGEN` 폴더로 이동
3. `2_2_2_preprocess.py` 파일 내 데이터 경로와 caption 수정 여부를 지정한 후 `GLIGEN` 폴더로 이동
4. `new_cpation`이 `True`일 때 caption을 문장형식으로 변환
```python
path = Path('ORIGIN_DATA_PATH')
new_caption = True
```
5. `2_2_2_preprocess.py` 실행
```bash
python 2_2_2_preprocess.py
```

## Train
1. `custom_dataset.py` 파일을 `GLIGEN/dataset` 폴더로 이동
2. `GLIGEN/dataset/catalog.py` 파일에 `__init__` 함수 안에 아래 내용 추가
```python
self.CustomTrain = {
    "target": "dataset.custom_dataset.CustomDataset",
    "train_params": dict(
        dataset_path=Path(ROOT),
    ),
  }
```
3.  `GLIGEN/configs` 폴더 안에 `custom_config.yaml` 추가
4.  다음 명령어 실행
```bash
python main.py --OUTPUT_ROOT={SAVE_PATH} --yaml_file=configs/custom_config.yaml --DATA_ROOT={DATA_PATH} --name={EXP_NAME} --total_iters={ITERATION_NUM}
```

## Inference
1. `GLIGEN/gligen_inference.py`에 `meta_list`에 값을 알맞게 넣어서 실행
