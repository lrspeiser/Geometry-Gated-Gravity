@echo off
setlocal
python -m gravity_learn.train.train_o2 --cfg gravity_learn\configs\o2_default.yaml --mode sr
endlocal
