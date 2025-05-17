import subprocess
import sys
import os
import time

# Пути до интерпретаторов виртуальных окружений (пример для Unix-подобных систем)
venv1_python = 'venv/bin/python'
venv2_python = 'venv/bin/python'
venv3_python = 'venv/bin/python'

# Команды запуска uvicorn для серверов
cmd1 = [venv1_python, '-m', 'uvicorn', 'app:app', '--port', '8000', '--reload']
cmd2 = [venv2_python, '-m', 'uvicorn', 'app:app', '--port', '8001', '--reload']

# Команда запуска бота
cmd3 = [venv3_python, 'bot.py']

# Запускаем процессы
p1 = subprocess.Popen(cmd1, cwd='boxmot')
print("Server 1 started on port 8000")

p2 = subprocess.Popen(cmd2, cwd='openpifpaf')
print("Server 2 started on port 8001")

p3 = subprocess.Popen(cmd3, cwd='bot')
print("Bot started")

try:
    # Ждем пока процессы не завершатся (или делаем паузу, чтобы скрипт не завершился сразу)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping all processes...")
    p1.terminate()
    p2.terminate()
    p3.terminate()
