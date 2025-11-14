@echo off
REM Скрипт для обучения с кэшем на диске D
echo ========================================
echo Настройка кэша Hugging Face на диск D
echo ========================================

REM Установить переменные окружения
set HF_HOME=D:\huggingface_cache
set HF_HUB_CACHE=D:\huggingface_cache\hub

REM Создать директории если не существуют
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

echo Кэш установлен на: %HF_HOME%
echo HF_HUB_CACHE: %HF_HUB_CACHE%
echo.

echo Запуск обучения...
python scripts/train_lora.py --person Mikassa

pause

