import json
import os
from pathlib import Path
import random
from typing import List, Dict, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import torch
from sklearn.metrics import precision_score, recall_score

if TYPE_CHECKING:
    from typing import Callable, ParamSpec

    P = ParamSpec("P")

    from src.data_generator import RADARDataGenerator
    from src.visualizer import RADARVisualizer
    from src.gallery import RADARGallery

    DataGeneratorFabric = Callable[P, "RADARDataGenerator"] | None
    VisualizerFabric = Callable[P, "RADARVisualizer"] | None
    GalleryFabric = Callable[P, "RADARGallery"] | None


class RADARDetector:
    """Основной класс для детекции целей на РЛИ"""

    def __init__(
        self,
        data_generator_fabric: "DataGeneratorFabric" = None,
        visualizer_fabric: "VisualizerFabric" = None,
        gallery_fabric: "GalleryFabric" = None,
        model_path: str | None = None,
    ) -> None:
        assert visualizer_fabric is not None, "Visualizer is required"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.load_model(model_path)
        self.data_generator_fabric = data_generator_fabric
        self.visualizer = visualizer_fabric()
        self.gallery_fabric = gallery_fabric
        self.metrics = {}

    def load_model(self, model_path: str | None = None) -> YOLO:
        """Загрузка или создание модели YOLO"""
        if model_path and Path(model_path).exists():
            model = YOLO(model_path)
            print(f"Загружена модель из {model_path}")
        else:
            # Создание новой модели YOLOv8
            model = YOLO("yolov8n.yaml")
            print("Создана новая модель YOLOv8")

        return model.to(self.device)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка РЛИ"""
        # Нормализация 16-битного изображения
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0

        # Приведение к формату [0, 1]
        image = np.clip(image, 0, 1)

        return image

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> tuple[List, List]:
        """Детекция целей на РЛИ"""
        processed_image = self.preprocess_image(image)

        # Преобразование для YOLO (3-канальное изображение)
        if len(processed_image.shape) == 2:
            processed_image = np.stack([processed_image] * 3, axis=-1)

        # Выполнение детекции
        results = self.model(
            processed_image,
            conf=confidence_threshold,
            imgsz=256
        )
        if len(results) == 0:
            return tuple(), tuple()

        bboxes = []
        confidence_scores = []

        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                # Координаты в формате YOLO [x_center, y_center, width, height]
                bbox = box.xywhn.cpu().numpy()[0].tolist()
                class_id = int(box.cls.cpu().numpy()[0])
                confidence = box.conf.cpu().numpy()[0]

                bboxes.append([class_id] + bbox)
                confidence_scores.append(confidence)

        return bboxes, confidence_scores

    def evaluate_on_synthetic_data(
        self, num_samples: int = 50, use_gallery: bool = True
    ) -> Dict:
        """Оценка модели на синтетических данных"""
        generator = self.data_generator_fabric()

        all_predictions = []
        all_ground_truth = []
        confidence_scores = []

        # Данные для галереи
        gallery_images = []
        gallery_true_bboxes = []
        gallery_pred_bboxes = []
        gallery_conf_scores = []
        gallery_params = []
        gallery_titles = []

        for i in range(num_samples):
            # Генерация синтетического примера
            num_targets = random.randint(1, 6)
            image, true_bboxes, params = generator.generate_sample(num_targets)

            # Детекция
            pred_bboxes, conf_scores = self.detect(image)

            # Сбор статистики
            all_predictions.extend([bbox[0] for bbox in pred_bboxes])
            all_ground_truth.extend([bbox[0] for bbox in true_bboxes])
            confidence_scores.extend(conf_scores)

            # Сохранение для галереи
            # Сохраняем ~10 примеров для галереи
            if i % max(1, num_samples // 10) == 0:
                gallery_images.append(image)
                gallery_true_bboxes.append(true_bboxes)
                gallery_pred_bboxes.append(pred_bboxes)
                gallery_conf_scores.append(conf_scores)
                gallery_params.append(params)
                gallery_titles.append(f"Пример {i}")

        # Показать галерею если нужно
        if use_gallery and gallery_images:
            gallery = self.gallery_fabric(
                gallery_images,
                gallery_true_bboxes,
                gallery_pred_bboxes,
                gallery_params,
                gallery_titles,
            )
            gallery.show()

        # Вычисление метрик
        if all_predictions and all_ground_truth:
            precision = precision_score(
                all_ground_truth,
                all_predictions[: len(all_ground_truth)],
                average="macro",
                zero_division=0,
            )
            recall = recall_score(
                all_ground_truth,
                all_predictions[: len(all_ground_truth)],
                average="macro",
                zero_division=0,
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Упрощенная матрица ошибок
            confusion_matrix = np.zeros((2, 2))
            for true, pred in zip(
                all_ground_truth, all_predictions[: len(all_ground_truth)]
            ):
                confusion_matrix[true, pred] += 1

            self.metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "map50": (precision + recall) / 2,  # Упрощенный mAP
                "confusion_matrix": confusion_matrix,
                "confidence_scores": confidence_scores,
                "class_distribution": {
                    "target": sum(1 for x in all_predictions if x == 0),
                    "false_target": sum(1 for x in all_predictions if x == 1),
                },
            }

        return self.metrics

    def evaluate_on_saved_dataset(
        self,
        dataset_path: str = "radar_dataset"
    ) -> dict:
        """Оценка модели на сохраненном датасете"""
        if not os.path.exists(dataset_path):
            print(f"Датасет {dataset_path} не найден!")
            return {}

        # Загрузка информации о датасете
        dataset_info_path = os.path.join(dataset_path, "dataset_info.json")
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)

        all_predictions = []
        all_ground_truth = []
        confidence_scores = []

        gallery_images = []
        gallery_true_bboxes = []
        gallery_pred_bboxes = []
        gallery_conf_scores = []
        gallery_params = []
        gallery_titles = []

        for sample_info in dataset_info["samples"]:
            # Загрузка изображения
            image_path = os.path.join(
                dataset_path, "images", sample_info["image_filename"]
            )
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # Загрузка аннотаций
            annotation_path = os.path.join(
                dataset_path, "annotations", sample_info["annotation_filename"]
            )
            true_bboxes = []
            with open(annotation_path, "r") as f:
                for line in f:
                    bbox = list(map(float, line.strip().split()))
                    true_bboxes.append(bbox)

            # Загрузка параметров
            params_path = os.path.join(
                dataset_path, "annotations", sample_info["params_filename"]
            )
            with open(params_path, "r") as f:
                params = json.load(f)

            # Детекция
            pred_bboxes, conf_scores = self.detect(image)

            # Сбор статистики
            all_predictions.extend([bbox[0] for bbox in pred_bboxes])
            all_ground_truth.extend([bbox[0] for bbox in true_bboxes])
            confidence_scores.extend(conf_scores)

            # Сохранение для галереи (первые 10 примеров)
            if len(gallery_images) < 10:
                gallery_images.append(image)
                gallery_true_bboxes.append(true_bboxes)
                gallery_pred_bboxes.append(pred_bboxes)
                gallery_conf_scores.append(conf_scores)
                gallery_params.append(params)
                gallery_titles.append(sample_info["image_filename"])

        # Показать галерею
        if gallery_images:
            gallery = self.gallery_fabric(
                gallery_images,
                gallery_true_bboxes,
                gallery_pred_bboxes,
                gallery_params,
                gallery_titles,
            )
            gallery.show()

        # Вычисление метрик
        if all_predictions and all_ground_truth:
            precision = precision_score(
                all_ground_truth,
                all_predictions[: len(all_ground_truth)],
                average="macro",
                zero_division=0,
            )
            recall = recall_score(
                all_ground_truth,
                all_predictions[: len(all_ground_truth)],
                average="macro",
                zero_division=0,
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            confusion_matrix = np.zeros((2, 2))
            for true, pred in zip(
                all_ground_truth, all_predictions[: len(all_ground_truth)]
            ):
                confusion_matrix[true, pred] += 1

            self.metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "map50": (precision + recall) / 2,
                "confusion_matrix": confusion_matrix,
                "confidence_scores": confidence_scores,
                "class_distribution": {
                    "target": sum(1 for x in all_predictions if x == 0),
                    "false_target": sum(1 for x in all_predictions if x == 1),
                },
            }

        return self.metrics

    def visualize_detection(
        self,
        image: np.ndarray,
        true_bboxes: List,
        pred_bboxes: List,
        params: Dict,
        title: str = "",
    ):
        """Визуализация результатов детекции с параметрами"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Исходное изображение
        if image.dtype == np.uint16:
            img_display = (image / 256).astype(np.uint8)
        else:
            img_display = image.copy()

        ax1.imshow(img_display, cmap="gray")
        ax1.set_title("Исходное РЛИ")
        ax1.axis("off")

        # Обнаруженные цели с тонкими рамками без текста
        detection_visualization = self.visualizer.draw_bboxes(
            image,
            pred_bboxes
        )
        ax2.imshow(cv2.cvtColor(detection_visualization, cv2.COLOR_BGR2RGB))
        ax2.set_title("Обнаруженные цели")
        ax2.axis("off")

        # Таблица параметров
        self.visualizer.create_params_table(params, ax3)

        if title:
            plt.suptitle(title, fontsize=16)

        plt.tight_layout()
        plt.show()

    def process_user_image(self, image_path: str):
        """Обработка пользовательского изображения"""
        if Path(image_path).exists():
            # Загрузка изображения
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print("Ошибка загрузки изображения")
                return

            # Детекция
            bboxes, confidence_scores = self.detect(image)

            # Создание параметров для отображения
            params = {
                "background_type": "user_image",
                "speckle_intensity": 0.0,
                "num_targets": len(bboxes),
                "image_size": image.shape[:2],
                "true_targets": sum(1 for b in bboxes if b[0] == 0),
                "false_targets": sum(1 for b in bboxes if b[0] == 1),
            }

            # Визуализация
            self.visualize_detection(
                image,
                [],
                bboxes,
                params,
                "Результаты детекции"
            )

            print(f"Обнаружено объектов: {len(bboxes)}")
            for i, (bbox, conf) in enumerate(zip(bboxes, confidence_scores)):
                class_name = "target" if bbox[0] == 0 else "false_target"
                print(f"Объект {i+1}: {class_name}, уверенность: {conf:.3f}")
        else:
            print("Файл не найден")
