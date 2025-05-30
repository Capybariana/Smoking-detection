# 🚬 Smoking Detection – Система детекции курения на видео

**Куратор:** Беспалов Сергей
**Участники:** Каратеев Георгий, Новикова Полина, Чернодоля Валерия

## 📌 Описание проекта

Цель проекта — автоматическое обнаружение жестов, связанных с курением, на видеозаписях с помощью методов машинного обучения и компьютерного зрения.
Разработка может использоваться для контроля соблюдения правил в общественных местах, мониторинга поведения или создания аналитических систем.

## 🔍 Проблематика

* Курение в неположенных местах — частая проблема в общественных учреждениях.
* Ручной мониторинг требует больших затрат времени и ресурсов.
* Необходима система, способная эффективно обрабатывать видеопотоки и обнаруживать случаи курения автоматически.

## 🧠 Решение

Система анализирует видео покадрово:

* Используется OpenPifPaf для детекции людей и извлечения их ключевых точек.
* Поверх этого применяется обученная модель классификации жестов, указывающих на курение.
* При обнаружении соответствующих жестов на кадре отображаются bounding boxes.
* Обработанное видео сохраняется в отдельный файл.

## 🗂 Используемый датасет

Для обучения модели использовался приватный датасет, собранный с камер наблюдения на территории НГУ.
Он содержит видео с курением и без него.

## 🛠 Установка

Python версии **>=3.9 и <=3.12** обязателен.

### 🔧 Установка компонентов

```bash
git clone https://github.com/Capybariana/Smoking-detection.git
cd Smoking-detection
```

#### 🧪 .env:
```
DB_URL=postgresql+asyncpg://user:password@localhost:5432/your_db_name
BOT_TOKEN=<TOKEN>
TELEGRAM_API_URL=<IP>:<PORT>/bot
```

#### 📦 Бот:

```bash
cd bot
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
git clone --recursive https://github.com/tdlib/telegram-bot-api.git
cd telegram-bot-api
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo cmake --build . --target install
```
#### 🗄️ БД:

```bash
python init_db.py
```
`❗ Убедитесь, что база данных your_db_name уже создана в PostgreSQL. Скрипт init_db.py создаёт только таблицы, но не саму базу.❗`

#### 🎯 BoxMOT:

```bash
cd boxmot
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
git clone https://github.com/mikel-brostrom/boxmot.git
cd boxmot
pip install .[yolo]
cd ..
pip install -r requirements.txt
```

#### 🕺 OpenPifPaf:
* ❗ОБЯЗАТЕЛЬНО PYTHON 3.9❗
```bash
cd openpifpaf
python3.9 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Запуск

* Склонируйте репозиторий (если необходимо) и перейдите в директорию проекта.
* Убедитесь, что все зависимости установлены.

Пример кода для запуска инференса в python
```bash
import bot.run_pipeline

if __name__ == "__main__":
    video_path = "input.mp4"  # имя файла в папке shared/videos
    final_output = run_full_pipeline(video_path)
    print("🎬 Final processed video saved at:", final_output)
```

Пример команды для запуска через Telegram Бота:

`Запуск Local Telegram API, для обхода ограничения на загрузку видео в 20 МБ.`
```bash
cd bot
./telegram-bot-api/build/telegram-bot-api \
  --local \
  --api-id=<ВАШ_ID> \
  --api-hash=<ВАШ_HASH> \
  --dir=<ПАПКА_БОТА> \
  --http-port=<ПОРТ> \
  --http-ip-address=127.0.0.1 \
  --log=bot_api.log \
  --verbosity=4

```

Аргументы:

* `--local` — запуск API в локальном режиме (без обращения к серверам Telegram).

* `--api-id` — ваш api_id, который можно получить на my.telegram.org.

* `--api-hash` — ваш api_hash, полученный с того же сайта.

* `--dir` — путь к директории для хранения базы данных Telegram (например, ./data).

* `--http-port` — порт, на котором API будет доступен (например, 8081).

* `--http-ip-address` — IP-адрес, по которому API будет слушать запросы (например, 127.0.0.1).

* `--log` (необязательный) — путь к файлу логов (по умолчанию лог пишется в консоль).

* `--verbosity` (необязательный) — уровень логирования (от 0 до 1024; рекомендуется 4–10 для отладки).

## ❗После успешного Telegram API запускаем бота и все необходимые модули.

```bash
cd Smoking-detection
python3 run_all.py  # НУЖНО ИМЕТЬ ЗАПУЩЕННЫЙ API сервер !!! ПО ГАЙДУ ВЫШЕ
```

## 📈 Возможности для доработки

* 📊 Улучшение модели: дообучение на более разнообразных сценах и ракурсах для повышения точности обнаружения.
* 🎥 Реалтайм обработка: добавить поддержку потокового видео с камеры в реальном времени.
* ⚠️ Обработка ошибок: внедрение логирования и исключения для недопустимых форматов.
* 🧩 Интеграция: возможность интеграции в существующие системы видеонаблюдения.
* 📊 Аналитика: сбор статистики по количеству случаев курения и визуализация по времени/месту.


## 📬 Обратная связь

По всем вопросам и предложениям — пишите в Telegram `@TastyToaster` или создавайте Issue на GitHub.
