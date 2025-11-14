@echo off
REM Скрипт для установки PyTorch nightly для RTX 5060 Ti
echo ======================================================================
echo УСТАНОВКА PYTORCH NIGHTLY ДЛЯ RTX 5060 Ti (sm_120)
echo ======================================================================
echo.

REM Активировать виртуальное окружение если есть
if exist venv\Scripts\activate.bat (
    echo Активация виртуального окружения...
    call venv\Scripts\activate.bat
)

REM Обновить pip
echo Обновление pip...
python -m pip install --upgrade pip

echo.
echo Удаление старой версии PyTorch...
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo Установка PyTorch nightly с CUDA 12.4+...
echo (Это может занять несколько минут и ~3-4 GB места)
echo.
echo ВАЖНО: Убедитесь, что на диске C свободно минимум 5 GB!
echo.
pause

echo Устанавливаем torch (это займет больше всего времени)...
echo (Предупреждения о зависимостях - это нормально для nightly сборок)
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

REM Проверяем, что torch действительно установился (игнорируем предупреждения pip о зависимостях)
REM Важно: проверяем импорт, а не ERRORLEVEL от pip, так как pip может вернуть ошибку из-за предупреждений
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ОШИБКА: Не удалось установить torch
    echo Возможные причины:
    echo 1. Недостаточно места на диске (нужно минимум 5 GB)
    echo 2. Проблемы с интернетом
    echo.
    echo См. docs/PYTORCH_SPACE_ISSUE.md для решения проблем
    pause
    exit /b 1
)

echo torch установлен успешно!
echo (Если pip показал предупреждения о зависимостях - это нормально для nightly сборок)

echo.
echo Устанавливаем torchvision...
echo (Предупреждения о зависимостях - это нормально для nightly сборок)
python -m pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps

REM Проверяем, что torchvision установился
python -c "import torchvision" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ПРЕДУПРЕЖДЕНИЕ: torchvision не установился, пробуем принудительную переустановку...
    python -m pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps --force-reinstall
)
python -c "import torchvision" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo torchvision установлен успешно!
) else (
    echo.
    echo ВНИМАНИЕ: torchvision не удалось установить автоматически
    echo Попробуйте установить вручную: pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps
    echo.
)

echo.
echo Попытка установить torchaudio (может быть недоступен в nightly)...
python -m pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 2>nul || echo torchaudio недоступен в nightly сборках, пропускаем...

echo.
echo ======================================================================
echo ПРОВЕРКА УСТАНОВКИ
echo ======================================================================
python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ======================================================================
echo УСТАНОВКА ЗАВЕРШЕНА
echo ======================================================================
pause

