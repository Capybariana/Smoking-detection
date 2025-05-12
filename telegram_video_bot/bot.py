import os
import asyncio
from uuid import uuid4
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from database import Session, Video

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

os.makedirs("videos", exist_ok=True)

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.video.get_file()
    filename = f"videos/{uuid4()}.mp4"
    await file.download_to_drive(filename)

    async with Session() as session:
        session.add(Video(file_path=filename))
        await session.commit()

    await update.message.reply_text("Видео получено. Ждём 10 секунд...")
    await asyncio.sleep(10)

    await context.bot.send_video(chat_id=update.message.chat_id, video=open(filename, 'rb'))

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.run_polling()

if __name__ == '__main__':
    main()
