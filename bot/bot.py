import os
import asyncio
from uuid import uuid4
from dotenv import load_dotenv
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters, CommandHandler

from database import Session, Video
from collections import deque
import logging

from run_pipeline import run_full_pipeline 


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("TELEGRAM_API_URL")



# Очередь для обработки видео
queue = deque()
processing = False # Флаг: сейчас обрабатывается видео

# --- Обработка очереди видео ---
async def process_queue(context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает видео из очереди по одному."""
    global processing
    if processing or not queue:
        # Если уже обрабатывается или очередь пуста, выходим
        return

    processing = True
    # Используем update и original_filename из очереди
    queue_item = queue.popleft()
    update = queue_item[0]
    original_filename = queue_item[1]

    chat_id = update.message.chat_id

    # Уведомление пользователя о начале обработки
    processing_message = await context.bot.send_message(chat_id=chat_id, text="Настала ваша очередь. Видео принято в обработку. Пожалуйста, ожидайте.")
    logger.info(f"Начата обработка файла из очереди: {original_filename}")

    video_entry = None # Инициализируем запись базы данных
    processed_output_path = None # Инициализируем путь к обработанному файлу

    try:
        # 1. Сохранение записи в базу данных
        async with Session() as session:
            video_entry = Video(file_path=original_filename, status="processing") # Добавьте статус
            session.add(video_entry)
            await session.commit()
            


        # 2. Запуск обработки видео в отдельном потоке
        logger.info(f"Запуск run_full_pipeline для {original_filename}")
        processed_output_path = await asyncio.to_thread(run_full_pipeline, original_filename)
        logger.info(f"run_full_pipeline завершен для {original_filename}, результат: {processed_output_path}")

        # 3. Отправка обработанного видео обратно пользователю
        if processed_output_path and os.path.exists(processed_output_path):
            logger.info(f"Отправка обработанного видео: {processed_output_path}")
            try:
                with open(processed_output_path, 'rb') as video_file:
                     # !!! Явно указываем таймауты для send_video !!!
                     # 600 секунд = 10 минут. Настройте по необходимости.
                     await context.bot.send_video(
                         chat_id=chat_id,
                         video=video_file,
                         read_timeout=600.0,   # Таймаут на чтение ответа
                         write_timeout=600.0  # Таймаут на отправку данных
                     )
                logger.info(f"Обработанное видео {processed_output_path} успешно отправлено.")

                # 4. Обновление статуса в базе данных после успешной отправки
                if video_entry:
                    async with Session() as session:
                        # Обновить статус по file_path:
                        await session.execute(Video.__table__.update().where(Video.file_path == original_filename).values(status="completed"))
                        await session.commit()

                # Удаляем сообщение "Пожалуйста, ожидайте"
                try:
                    await processing_message.delete()
                except Exception as delete_error:
                    logger.warning(f"Не удалось удалить сообщение об обработке: {delete_error}")

                await context.bot.send_message(chat_id=chat_id, text="Обработка завершена, видео отправлено.")


            except Exception as send_error:
                 logger.error(f"Ошибка при отправке видео {processed_output_path}: {send_error}", exc_info=True)
                 await context.bot.send_message(chat_id=chat_id, text=f"Произошла ошибка при отправке обработанного видео: {send_error}")

                 # 5. Обновление статуса при ошибке отправки
                 if video_entry:
                     async with Session() as session:
                         await session.execute(Video.__table__.update().where(Video.file_path == original_filename).values(status="sending_failed", error_message=str(send_error)[:255]))
                         await session.commit()


        else:
            error_msg = f"Обработка завершилась, но результат не найден: {processed_output_path}"
            logger.error(error_msg)
            await context.bot.send_message(chat_id=chat_id, text=f"Произошла ошибка обработки: {error_msg}")
            # Обновление статуса при отсутствии результата
            if video_entry:
                 async with Session() as session:
                     await session.execute(Video.__table__.update().where(Video.file_path == original_filename).values(status="processing_failed_no_output", error_message=error_msg))
                     await session.commit()


    except Exception as e:
        # Обработка любых других ошибок на этапе обработки (кроме ошибок отправки, которые ловятся выше)
        logger.error(f"Ошибка обработки видео {original_filename}: {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"Произошла ошибка при обработке вашего видео: {e}")

        # Обновление статуса при ошибке обработки
        if video_entry:
             async with Session() as session:
                 await session.execute(Video.__table__.update().where(Video.file_path == original_filename).values(status="processing_failed", error_message=str(e)[:255]))
                 await session.commit()


    finally:
        processing = False
        logger.info(f"Обработка файла {original_filename} завершена (или произошла ошибка).")

        # Очистка временных файлов
        if os.path.exists(original_filename):
           try:
               os.remove(original_filename)
               logger.info(f"Removed original file: {original_filename}")
           except OSError as e:
               logger.error(f"Error removing original file {original_filename}: {e}")
        if processed_output_path and os.path.exists(processed_output_path):
           try:
               os.remove(processed_output_path)
               logger.info(f"Removed processed file: {processed_output_path}")
           except OSError as e:
               logger.error(f"Error removing processed file {processed_output_path}: {e}")

        await process_queue(context)


# --- Определение пути сохранения файла ---
# Синхронная функция, будет запускаться в потоке
def extract_custom_path(update: Update) -> str:
    """Ищет пользовательскую директорию сохранения в тексте/подписи.
       Возвращает безопасный полный путь или базовую директорию при ошибке/отсутствии команды.
       Сообщения об ошибках отправляет вызывающая асинхронная функция.
    """
    base_dir = os.path.normpath("../shared/videos")
    text = update.message.caption or update.message.text or ""

    # Проверяем, начинается ли текст именно с команды /save_dir с пробелом
    if text.strip().startswith("/save_dir "):
        custom_path_input = text.strip().replace("/save_dir ", "", 1).strip() # Удаляем команду один раз и обрезаем пробелы

        if not custom_path_input:
             # Пользователь ввел только /save_dir без пути
             logger.warning("Received /save_dir without a path.")
             os.makedirs(base_dir, exist_ok=True) # Убедиться, что базовая папка есть
             return base_dir # Вызывающая функция отправит сообщение об ошибке

        # Очистка и проверка компонентов пути для безопасности
        # Разбиваем путь, удаляем пустые части, '.' и '..'
        safe_path_components = [
            comp for comp in os.path.normpath(custom_path_input).split(os.sep)
            if comp not in ('', '.', '..')
        ]

        if not safe_path_components:
            # Если после очистки путь пуст (был что-то вроде ../../../)
            logger.warning(f"Unsafe path components provided: {custom_path_input}. Using base directory.")
            os.makedirs(base_dir, exist_ok=True) # Убедиться, что базовая папка есть
            return base_dir # Вызывающая функция отправит сообщение об ошибке

        # Объединяем безопасные компоненты с базовой директорией
        full_path = os.path.join(base_dir, *safe_path_components)
        logger.info(f"Attempting to use custom path: {full_path} from input: {custom_path_input}")

        try:
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Successfully ensured directory exists: {full_path}")
            return full_path
        except OSError as e:
             # Ошибка при создании директории (например, недопустимые символы в имени файла/папки)
             logger.error(f"Error creating directory {full_path} from input {custom_path_input}: {e}")
             os.makedirs(base_dir, exist_ok=True)
             return base_dir
        except Exception as e:
             # Любая другая неожиданная ошибка
             logger.error(f"Unexpected error in extract_custom_path for input {custom_path_input}: {e}", exc_info=True)
             os.makedirs(base_dir, exist_ok=True) # Убедиться, что базовая папка есть
             return base_dir # Вызывающая функция отправит сообщение об ошибке


    # Если команда /save_dir не использовалась, используем базовую директорию
    os.makedirs(base_dir, exist_ok=True) # Убедиться, что базовая папка есть
    return base_dir


# --- Обработчик получения видео ---
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие видео или видео-документы."""
    logger.info(f"Received potential video from user {update.effective_user.id} in chat {update.effective_chat.id}")


    message = update.message
    media = None

    # Проверяем, является ли сообщение видео или документом с MIME-типом видео
    if message.video:
        media = message.video
        logger.info(f"Detected message.video with file_id: {media.file_id}, size: {media.file_size}")
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("video/"):
         media = message.document
         logger.info(f"Detected message.document (video) with file_id: {media.file_id}, size: {media.file_size}, mime_type: {media.mime_type}")
    else:
        logger.warning("Received message filtered as video but not video/document video type.")
        await message.reply_text("⚠️ Пожалуйста, отправьте видео (MP4).")
        return

    # Проверка размера файла до скачивания из Telegram (media уже содержит file_size)
    MAX_FILE_SIZE_MB = 50
    if media.file_size is None or media.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await message.reply_text(f"⚠️ Видео слишком большое (> {MAX_FILE_SIZE_MB} МБ) или не удалось определить его размер.")
        logger.warning(f"Rejected video file_id: {media.file_id} due to size {media.file_size}.")
        return


    # Таймаут на получение объекта файла от Telegram (обычно быстрый)
    GET_FILE_TIMEOUT = 10.0 # Таймаут в секундах
    tg_file = None # Инициализируем tg_file

    try:
        # Получаем объект файла от Telegram
        tg_file = await asyncio.wait_for(media.get_file(), timeout=GET_FILE_TIMEOUT)
        logger.info(f"Got file object from Telegram for file_id: {media.file_id}")

    except asyncio.TimeoutError:
         logger.error(f"Timeout getting file object from Telegram for file_id: {media.file_id}")
         await message.reply_text(f"⚠️ Не удалось получить информацию о файле от Telegram за {GET_FILE_TIMEOUT} секунд. Попробуйте еще раз.")
         return
    except BadRequest as e:
        # Ошибка от Telegram (например, неверный file_id, хотя media.get_file обычно работает)
        logger.error(f"BadRequest getting file from Telegram for file_id {media.file_id}: {e}", exc_info=True)
        await message.reply_text(f"⚠️ Не удалось получить файл от Telegram: {e}")
        return
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении файла от Telegram для file_id {media.file_id}: {e}", exc_info=True)
        await message.reply_text(f"⚠️ Неожиданная ошибка при получении файла: {e}")
        return

    # Определяем путь сохранения на диске (синхронная операция, выполняем в потоке)
    # extract_custom_path может вернуть базовую директорию и не отправить сообщение,
    # если ошибка парсинга пути. Мы отправим сообщение здесь, основываясь на результате extract_custom_path.
    save_dir = await asyncio.to_thread(extract_custom_path, update)

    # Проверяем, был ли запрос на custom_path и успешно ли он отработал
    user_text = update.message.caption or update.message.text or ""
    base_dir = os.path.normpath("../shared/videos")

    # Отправляем сообщение пользователю о том, куда будет сохранен файл
    # на основе анализа того, что вернула extract_custom_path и что ввел пользователь
    if user_text.strip().startswith("/save_dir "):
         if save_dir == base_dir:
              # extract_custom_path вернула базовую директорию, хотя пользователь пытался указать свою.
              # extract_custom_path уже залогировала причину. Отправим сообщение пользователю.
              await message.reply_text(f"⚠️ Не удалось использовать указанный путь сохранения. Видео будет сохранено в базовой папке.")
         else:
              # custom_path успешно определен и создан (или это была базовая папка, если custom_path_input был пуст)
              await message.reply_text(f"Видео успешно получено!")
    else:
         # Команда /save_dir не использовалась, видео сохраняется в базовой папке
         await message.reply_text(f"Видео успешно получено!")


    # Генерируем уникальное имя файла в выбранной директории
    # Используем оригинальное расширение, если доступно, иначе mp4
    original_extension = os.path.splitext(media.file_name or "")[1] or ".mp4"
    filename = os.path.join(save_dir, f"{uuid4()}{original_extension}")


    # Таймаут на скачивание файла на диск из Telegram
    DOWNLOAD_TO_DISK_TIMEOUT = 300.0
    try:
        # Скачиваем файл на диск
        logger.info(f"Downloading file {media.file_id} to {filename} (size: {media.file_size} bytes)")
        await asyncio.wait_for(tg_file.download_to_drive(filename), timeout=DOWNLOAD_TO_DISK_TIMEOUT)
        logger.info(f"Successfully downloaded file to {filename}")

    except asyncio.TimeoutError:
        logger.error(f"Download to disk timed out for file {media.file_id} at {filename}")
        await message.reply_text(f"⚠️ Не удалось скачать файл на диск за {DOWNLOAD_TO_DISK_TIMEOUT} секунд.")
        # Очистить, если файл скачался частично
        if os.path.exists(filename):
            try:
                os.remove(filename)
                logger.info(f"Removed partial download: {filename}")
            except OSError as e:
                logger.error(f"Error removing partial download {filename}: {e}")
        return
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла на диск {filename}: {e}", exc_info=True)
        await message.reply_text(f"⚠️ Ошибка при скачивании файла на диск: {e}")
        # Очистить, если файл скачался частично
        if os.path.exists(filename):
            try:
                os.remove(filename)
                logger.info(f"Removed partial download after error: {filename}")
            except OSError as e:
                logger.error(f"Error removing partial download {filename} after error: {e}")
        return

    # Добавляем скачанный файл (его путь) и update в очередь обработки
    queue.append((update, filename))
    position = len(queue)
    await message.reply_text(f"Ваше видео добавлено в очередь. Место в очереди: {position}")
    logger.info(f"File {filename} added to queue. Current queue size: {position}")

    # Запускаем обработку очереди (она сама проверит, не занята ли)
    # Этот вызов не блокирует handle_video, хэндлер завершится сразу после этого.
    asyncio.create_task(process_queue(context))


# --- Обработчик команд (пример) ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    await update.message.reply_text("Привет! Отправь мне видео для обработки.")

# --- Главная функция ---
def main():
    print("ЗАПУСК БОТА")
    logger.info(f"TOKEN: {TOKEN}")
    logger.info(f"API_URL: {API_URL}")

    app = ApplicationBuilder().token(TOKEN).base_url(API_URL).build()

    
    app.add_handler(CommandHandler("start", start_command))

    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))



    logger.info("Бот запущен и ожидает команды или видео...")

    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)


if __name__ == '__main__':
    main()