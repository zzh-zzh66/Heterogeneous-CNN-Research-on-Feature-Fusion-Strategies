@echo off
REM Batch training script for all experiments
REM Experiment order:
REM   1. Single branch models (M1, M2, M3)
REM   2. Homo ensemble model (M4)
REM   3. Hetero fusion models with different lambda values (M5)

echo ============================================================
echo CNN Feature Fusion - CIFAR-10 Batch Training
echo ============================================================
echo.

cd /d "%~dp0src"

set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGFILE=..\results\batch_log_%TIMESTAMP%.txt

if not exist "..\results\M1_branch1" mkdir "..\results\M1_branch1"
if not exist "..\results\M2_branch2" mkdir "..\results\M2_branch2"
if not exist "..\results\M3_branch3" mkdir "..\results\M3_branch3"
if not exist "..\results\M4_homo_ensemble" mkdir "..\results\M4_homo_ensemble"
if not exist "..\results\M5_hetero_lambda0" mkdir "..\results\M5_hetero_lambda0"
if not exist "..\results\M5_hetero_lambda0p01" mkdir "..\results\M5_hetero_lambda0p01"
if not exist "..\results\M5_hetero_lambda0p05" mkdir "..\results\M5_hetero_lambda0p05"
if not exist "..\results\M5_hetero_lambda0p1" mkdir "..\results\M5_hetero_lambda0p1"
if not exist "..\results\M5_hetero_lambda0p5" mkdir "..\results\M5_hetero_lambda0p5"

echo Start time: %date% %time%
echo Log file: %LOGFILE%
echo.
echo ============================================================
echo [1/9] Train Single Branch1 (Texture)
echo ============================================================
echo.
python main.py --model single_b1 --save_dir ../results

echo.
echo ============================================================
echo [2/9] Train Single Branch2 (Edge)
echo ============================================================
echo.
python main.py --model single_b2 --save_dir ../results

echo.
echo ============================================================
echo [3/9] Train Single Branch3 (Semantic)
echo ============================================================
echo.
python main.py --model single_b3 --save_dir ../results

echo.
echo ============================================================
echo [4/9] Train Homo Ensemble
echo ============================================================
echo.
python main.py --model homo_ensemble --save_dir ../results

echo.
echo ============================================================
echo [5/9] Train Hetero Fusion (lambda=0)
echo ============================================================
echo.
python main.py --model hetero_fusion --lambda_orth 0 --save_dir ../results

echo.
echo ============================================================
echo [6/9] Train Hetero Fusion (lambda=0.01)
echo ============================================================
echo.
python main.py --model hetero_fusion --lambda_orth 0.01 --save_dir ../results

echo.
echo ============================================================
echo [7/9] Train Hetero Fusion (lambda=0.05)
echo ============================================================
echo.
python main.py --model hetero_fusion --lambda_orth 0.05 --save_dir ../results

echo.
echo ============================================================
echo [8/9] Train Hetero Fusion (lambda=0.1)
echo ============================================================
echo.
python main.py --model hetero_fusion --lambda_orth 0.1 --save_dir ../results

echo.
echo ============================================================
echo [9/9] Train Hetero Fusion (lambda=0.5)
echo ============================================================
echo.
python main.py --model hetero_fusion --lambda_orth 0.5 --save_dir ../results

echo.
echo ============================================================
echo All experiments completed!
echo End time: %date% %time%
echo ============================================================
echo.
echo See ../results directory for results
echo Training log saved to: %LOGFILE%

pause
