import os
from dotenv import load_dotenv
from sqlalchemy import String # Импортируйте String
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
# Импортируйте Mapped и mapped_column из sqlalchemy.orm (у вас уже есть)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()
DATABASE_URL = os.getenv("DB_URL")

engine = create_async_engine(DATABASE_URL, echo=False)
Session = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str]

    # --- Добавьте эти строки ---
    # Поле для статуса обработки (например: 'pending', 'processing', 'completed', 'failed')
    # default="pending" устанавливает начальное значение
    status: Mapped[str] = mapped_column(String(50), default="pending") # Укажите подходящую максимальную длину строки

    # Поле для сообщения об ошибке (может быть NULL)
    error_message: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    # --------------------------

    # Опционально, добавьте __repr__ для удобства отладки
    def __repr__(self) -> str:
        return f"Video(id={self.id!r}, file_path={self.file_path!r}, status={self.status!r}, error_message={self.error_message!r})"