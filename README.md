# vEpiNetV2

## Create a virtual environment with conda

```python
conda create -n vepinetv2 python=3.9
conda activate vepinetv2
pip install -r requirements.txt
```

## How to run
1.`vim train_config.py`
2.`python train.py`
3.`python eval.py --test_path test_set_csv_file_path --weight model_file_path`