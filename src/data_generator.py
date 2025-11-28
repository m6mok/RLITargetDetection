from enum import Enum
import random
from collections import Counter
from pathlib import Path

import numpy as np
import cv2

from src.utils.io import makedirs, write_image, write_json, write_text

RADAR_DATASET_PATH = Path("radar_dataset")


class TargetType(Enum):
    TARGET = "target"
    FALSE_TARGET = "false_target"


class RADARDataGenerator:
    """Генератор синтетических радиолокационных изображений"""

    def __init__(self, image_size: tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.background_types = ['sea', 'land', 'urban']

    def generate_speckle_noise(
        self,
        shape: tuple[int, int],
        intensity: float = 0.1
    ) -> np.ndarray:
        """Генерация спекл-шума"""
        noise = np.random.gamma(shape=1, scale=intensity, size=shape)
        return noise

    def generate_background(self, bg_type: str) -> np.ndarray:
        """Генерация фона различного типа"""
        h, w = self.image_size

        if bg_type == 'sea':
            # Морская поверхность с волнами
            background = np.zeros((h, w))
            for i in range(20):
                x = np.linspace(0, 4*np.pi, w)
                y = np.sin(x + i) * 0.1 + 0.2
                wave = np.tile(y, (h//20, 1))
                background[i*(h//20):(i+1)*(h//20), :] = wave
            background += np.random.normal(0, 0.05, (h, w))

        elif bg_type == 'land':
            # Земная поверхность с неровностями
            background = np.random.normal(0.3, 0.15, (h, w))
            # Добавляем крупные структуры
            for _ in range(8):
                cx, cy = random.randint(0, w), random.randint(0, h)
                size = random.randint(15, 40)
                background = cv2.circle(background, (cx, cy), size, 0.2, -1)

        else:  # urban
            # Урбанистический ландшафт
            background = np.random.normal(0.4, 0.2, (h, w))
            # Прямоугольные структуры (здания)
            for _ in range(12):
                x1, y1 = (
                    random.randint(0, w-20),
                    random.randint(0, h-20),
                )
                x2, y2 = (
                    x1 + random.randint(10, 25),
                    y1 + random.randint(10, 25),
                )
                background[y1:y2, x1:x2] += 0.3

        return np.clip(background, 0, 1)

    def generate_target(
        self,
        target_type: TargetType = TargetType.TARGET
    ) -> tuple[np.ndarray, list[int]]:
        """Генерация цели"""
        size = random.randint(8, 30)
        h, w = self.image_size

        # Случайная позиция
        x_center = random.randint(size, w - size)
        y_center = random.randint(size, h - size)

        # Создание цели
        target = np.zeros((size, size))

        if target_type == TargetType.TARGET:
            # Истинная цель - яркий объект с четкими границами
            if random.random() > 0.5:
                # Эллиптическая цель
                target = cv2.ellipse(
                    np.zeros((size, size)),
                    (size//2, size//2),
                    (size//2, size//3),
                    0,
                    0,
                    360,
                    0.8,
                    -1,
                )
            else:
                # Прямоугольная цель
                target = cv2.rectangle(
                    np.zeros((size, size)),
                    (1, 1),
                    (size-1, size-1),
                    0.8,
                    -1,
                )
        else:
            # Ложная цель - размытый или шумный объект
            target = np.random.normal(0.4, 0.2, (size, size))
            target = cv2.GaussianBlur(target, (3, 3), 0.5)

        bbox = [x_center - size//2, y_center - size//2, size, size]
        return target, bbox

    def generate_rfi_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Генерация радиочастотных помех"""
        h, w = image.shape
        result = image.copy()

        # Линейные артефакты (полосы)
        for _ in range(3):
            if random.random() > 0.7:
                row = random.randint(0, h-1)
                result[row, :] += random.uniform(0.1, 0.3)

        # Точечные артефакты
        for _ in range(8):
            if random.random() > 0.8:
                x, y = random.randint(0, w-1), random.randint(0, h-1)
                result[y, x] += random.uniform(0.2, 0.5)

        return np.clip(result, 0, 1)

    def generate_sample(
        self,
        num_targets: int = 3,
    ) -> tuple[np.ndarray, list[list], dict]:
        """Генерация одного примера РЛИ с возвратом параметров"""
        # Выбор типа фона
        bg_type = random.choice(self.background_types)
        image = self.generate_background(bg_type)

        # Добавление спекл-шума
        speckle_intensity = random.uniform(0.1, 0.2)
        speckle = self.generate_speckle_noise(
            self.image_size,
            speckle_intensity,
        )
        image += speckle

        bboxes = []
        labels = []
        targets_info = []

        # Добавление целей
        for _ in range(num_targets):
            target_type: TargetType = (
                TargetType.TARGET
                if random.random() > 0.3
                else TargetType.FALSE_TARGET
            )
            target_img, bbox = self.generate_target(target_type)

            x, y, w, h = bbox
            image[y:y+h, x:x+w] += target_img

            # Нормализованные координаты для YOLO
            x_center_norm = (x + w/2) / self.image_size[1]
            y_center_norm = (y + h/2) / self.image_size[0]
            w_norm = w / self.image_size[1]
            h_norm = h / self.image_size[0]

            class_id = 0 if target_type == TargetType.TARGET else 1
            bboxes.append(
                [class_id, x_center_norm, y_center_norm, w_norm, h_norm]
            )
            labels.append(target_type)

            # Сохраняем информацию о цели
            targets_info.append({
                'type': target_type,
                'bbox': bbox,
                'size': (w, h)
            })

        # Добавление помех
        image = self.generate_rfi_artifacts(image)

        # Нормализация в 16-битный формат
        image = (image * 65535).astype(np.uint16)

        target_count = Counter(target["type"] for target in targets_info)

        # Параметры генерации
        params = {
            'background_type': bg_type,
            'speckle_intensity': speckle_intensity,
            'num_targets': num_targets,
            'targets_info': targets_info,
            'image_size': self.image_size,
            'true_targets': target_count[TargetType.TARGET],
            'false_targets': target_count[TargetType.FALSE_TARGET],
        }

        return image, bboxes, params

    def generate_dataset(
        self,
        num_samples: int,
        save_path: Path = RADAR_DATASET_PATH,
    ) -> Path:
        """Генерация и сохранение набора РЛИ"""
        makedirs(save_path / "images")
        makedirs(save_path / "annotations")

        dataset_info = {
            'total_samples': num_samples,
            'image_size': self.image_size,
            'samples': []
        }

        for i in range(num_samples):
            num_targets = random.randint(1, 6)
            image, bboxes, params = self.generate_sample(num_targets)

            # Сохранение изображения
            image_filename = f"radar_{i:06d}.png"
            write_image(save_path / "images" / image_filename, image)

            # Сохранение аннотаций
            annotation_filename = f"radar_{i:06d}.txt"
            write_text(
                save_path / "annotations" / annotation_filename,
                "\n".join(
                    " ".join(str(x) for x in bbox)
                    for bbox in bboxes
                )
            )

            # Сохранение параметров
            params_filename = f"radar_{i:06d}_params.json"
            write_json(save_path / "annotations" / params_filename, params)

            dataset_info['samples'].append({
                'image_filename': image_filename,
                'annotation_filename': annotation_filename,
                'params_filename': params_filename,
                'num_targets': num_targets,
            })

        # Сохранение информации о датасете
        write_json(save_path / "dataset_info.json", dataset_info)

        return save_path
