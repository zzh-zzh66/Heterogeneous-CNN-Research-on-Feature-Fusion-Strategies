@echo off
REM Batch training script for M6 Hetero Fusion Coop experiments
REM Experiment order:
REM   1. M6:   lambda_coop=0.1,  lambda_aux=0.3  (default)
REM   2. M6_v2: lambda_coop=0.05, lambda_aux=0.3  (reduce coop loss)
REM   3. M6_v3: lambda_coop=0.1,  lambda_aux=0.15 (reduce aux loss)
REM   4. M6_v4: lambda_coop=0.0,  lambda_aux=0.3  (pure aux, no coop)

echo ============================================================
echo M6 Hetero Fusion Coop - Batch Training (4 experiments)
echo ============================================================
echo.

cd /d "%~dp0"

set SAVE_DIR=..\results
echo Results will be saved to: %SAVE_DIR%
echo.

echo ============================================================
echo [1/4] M6: lambda_coop=0.1, lambda_aux=0.3 (default)
echo ============================================================
echo.
python main.py --model hetero_fusion_coop --lambda_coop 0.1 --lambda_aux 0.3 --save_dir %SAVE_DIR%

echo.
echo ============================================================
echo [2/4] M6_v2: lambda_coop=0.05, lambda_aux=0.3
echo        (reduce coop reward, avoid forced collaboration)
echo ============================================================
echo.
python main.py --model hetero_fusion_coop --lambda_coop 0.05 --lambda_aux 0.3 --save_dir %SAVE_DIR%

echo.
echo ============================================================
echo [3/4] M6_v3: lambda_coop=0.1, lambda_aux=0.15
echo        (reduce aux loss, lighten single-branch dependency)
echo ============================================================
echo.
python main.py --model hetero_fusion_coop --lambda_coop 0.1 --lambda_aux 0.15 --save_dir %SAVE_DIR%

echo.
echo ============================================================
echo [4/4] M6_v4: lambda_coop=0.0, lambda_aux=0.3
echo        (pure aux loss, no coop constraint)
echo ============================================================
echo.
python main.py --model hetero_fusion_coop --lambda_coop 0.0 --lambda_aux 0.3 --save_dir %SAVE_DIR%

echo.
echo ============================================================
echo All M6 experiments completed!
echo ============================================================
echo.
echo Results saved to:
echo   - %SAVE_DIR%\M6_hetero_coop_c0p1_a0p3   (M6 default)
echo   - %SAVE_DIR%\M6_hetero_coop_c0p05_a0p3  (M6_v2)
echo   - %SAVE_DIR%\M6_hetero_coop_c0p1_a0p15  (M6_v3)
echo   - %SAVE_DIR%\M6_hetero_coop_c0p0_a0p3   (M6_v4)
echo.

pause
