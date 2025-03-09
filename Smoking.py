import argparse

import cv2
from ultralytics import YOLO


def process_video(input_video_path, output_video_path):
    """
    Обрабатывает видео с использованием модели YOLO и сохраняет результат в выходной файл.
    
    :param input_video_path: Путь к входному видеофайлу
    :param output_video_path: Путь для сохранения выходного видеофайла
    """
    # Загрузка модели
    model = YOLO('CigaretteDetect.pt')  # Используем модель, которую обучили

    # Открываем видео
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'vp80')  # VP8 codec для WebM
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Ошибка: Не удалось создать файл для записи {output_video_path}")
        cap.release()
        return

    print("Начинаем обработку видео...")
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Применение модели для предсказания на кадре с порогом уверенности 0.36
        results = model(frame, conf=0.36)

        frame_with_boxes = results[0].plot()

        out.write(frame_with_boxes)

    print("Обработка завершена.")

    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка видео с использованием модели YOLO и сохранение результата.")
    parser.add_argument("input_video", type=str, help="Путь к входному видеофайлу")
    parser.add_argument("output_video", type=str, help="Путь для сохранения выходного видеофайла")

    # Парсим аргументы
    args = parser.parse_args()

    # Вызываем функцию обработки видео
    process_video(args.input_video, args.output_video)