import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns


class RADARVisualizer:
    """Класс для визуализации РЛИ и результатов детекции"""

    def __init__(self):
        # Зеленый для целей, красный для ложных
        self.colors = [(0, 255, 0), (0, 0, 255)]

    def denormalize_bbox(
        self,
        bbox: list[float],
        img_shape: tuple[int, int]
    ) -> list[int]:
        """Денормализация bounding box"""
        class_id, x_center, y_center, width, height = bbox
        img_h, img_w = img_shape

        x_center_px = int(x_center * img_w)
        y_center_px = int(y_center * img_h)
        width_px = int(width * img_w)
        height_px = int(height * img_h)

        x1 = x_center_px - width_px // 2
        y1 = y_center_px - height_px // 2
        x2 = x_center_px + width_px // 2
        y2 = y_center_px + height_px // 2

        return [class_id, x1, y1, x2, y2]

    def draw_bboxes(self, image: np.ndarray, bboxes: list[list]) -> np.ndarray:
        """
        Отрисовка bounding boxes на изображении
        с тонкими рамкамибез текста
        """

        # Конвертация в 8-бит для OpenCV
        if image.dtype == np.uint16:
            img_viz = (image / 256).astype(np.uint8)
        else:
            img_viz = image.copy()

        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2BGR)

        for bbox in bboxes:
            class_id, x1, y1, x2, y2 = self.denormalize_bbox(bbox, image.shape)
            color = self.colors[int(class_id)]

            # Тонкая рамка (толщина 1) без текста
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), color, 1)

        return img_viz

    def create_params_table(self, params: dict, ax: plt.Axes):
        """Создание таблицы с параметрами РЛИ"""
        # Подготовка данных для таблицы
        table_data: tuple[list[str], ...] = (
            ("Параметр", "Значение"),
            ("Тип фона", params.get('background_type', 'N/A')),
            ("Интенсивность шума", f"{params.get('speckle_intensity', 0):.3f}"),
            ("Всего целей", str(params.get('num_targets', 0))),
            ("Истинные цели", str(params.get('true_targets', 0))),
            ("Ложные цели", str(params.get('false_targets', 0))),
            ("Размер изображения", f"{params.get('image_size', (0, 0))}"),
        )

        # Создание таблицы
        table = ax.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Стилизация таблицы
        for i, key in enumerate(table_data):
            for j, cell in enumerate(key):
                table[(i, j)].set_facecolor('lightblue' if i == 0 else 'white')
                table[(i, j)].set_edgecolor('black')

        ax.axis('off')
        ax.set_title('Параметры РЛИ', fontsize=12, pad=20)

    def plot_statistics(self, metrics: dict, save_path: str | None = None):
        """Визуализация статистики работы алгоритма"""
        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # График метрик
        metric_names = ('Precision', 'Recall', 'mAP@0.5', 'F1-Score')
        metric_values: tuple[str, ...] = (
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('map50', 0),
            metrics.get('f1_score', 0),
        )

        axes[0, 0].bar(
            metric_names,
            metric_values,
            color=('blue', 'green', 'red', 'purple'),
        )
        axes[0, 0].set_title('Метрики качества детекции')
        axes[0, 0].set_ylim(0, 1)

        # Матрица ошибок
        confusion_matrix = metrics.get('confusion_matrix', np.zeros((2, 2)))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Pred Target', 'Pred False'],
            yticklabels=['True Target', 'True False'],
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Матрица ошибок')

        # Распределение уверенности
        confidences = metrics.get('confidence_scores', [])
        if confidences:
            axes[1, 0].hist(confidences, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Распределение уверенности предсказаний')
            axes[1, 0].set_xlabel('Уверенность')
            axes[1, 0].set_ylabel('Частота')

        # Количество обнаружений по классам
        class_counts = metrics.get('class_distribution', {})
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            axes[1, 1].pie(
                counts,
                labels=classes,
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'],
            )
            axes[1, 1].set_title('Распределение обнаружений по классам')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
