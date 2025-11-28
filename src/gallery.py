from typing import TYPE_CHECKING, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import Button

if TYPE_CHECKING:
    from typing import Callable

    from src.visualizer import RADARVisualizer

    VisualizerFabric = Callable[[], "RADARVisualizer"] | None


class RADARGallery:
    """Класс для интерактивной галереи РЛИ"""

    def __init__(
        self,
        images: List[np.ndarray],
        true_bboxes: List[List],
        pred_bboxes: List[List],
        params: List[Dict],
        titles: List[str] = None,
        visualizer_fabric: "VisualizerFabric" = None,
    ):
        self.images = images
        self.true_bboxes = true_bboxes
        self.pred_bboxes = pred_bboxes
        self.params = params
        self.titles = titles or [f"Пример {i}" for i in range(len(images))]

        self.current_index = 0
        self.visualizer = visualizer_fabric()

    def show(self):
        """Показать интерактивную галерею"""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            1,
            3,
            figsize=(18, 6)
        )

        # Создаем кнопки
        ax_prev = plt.axes([0.1, 0.01, 0.1, 0.05])
        ax_next = plt.axes([0.8, 0.01, 0.1, 0.05])
        ax_save = plt.axes([0.45, 0.01, 0.1, 0.05])

        self.btn_prev = Button(ax_prev, "Предыдущий")
        self.btn_next = Button(ax_next, "Следующий")
        self.btn_save = Button(ax_save, "Сохранить")

        self.btn_prev.on_clicked(self.previous)
        self.btn_next.on_clicked(self.next)
        self.btn_save.on_clicked(self.save_current)

        self.update_plot()
        plt.tight_layout()
        plt.show()

    def update_plot(self):
        """Обновить отображение"""
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Текущие данные
        img = self.images[self.current_index]
        true_bbox = self.true_bboxes[self.current_index]
        # pred_bbox = self.pred_bboxes[self.current_index]
        param = self.params[self.current_index]
        title = self.titles[self.current_index]

        # Исходное изображение
        if img.dtype == np.uint16:
            img_display = (img / 256).astype(np.uint8)
        else:
            img_display = img.copy()

        self.ax1.imshow(img_display, cmap="gray")
        self.ax1.set_title("Исходное РЛИ")

        # Ground truth с тонкими рамками без текста
        gt_visualization = self.visualizer.draw_bboxes(img, true_bbox)
        self.ax2.imshow(cv2.cvtColor(gt_visualization, cv2.COLOR_BGR2RGB))
        self.ax2.set_title("Обнаруженные цели")

        # Таблица параметров
        self.visualizer.create_params_table(param, self.ax3)

        self.fig.suptitle(title, fontsize=14)
        plt.draw()

    def next(self, event):
        """Следующее изображение"""
        self.current_index = (self.current_index + 1) % len(self.images)
        self.update_plot()

    def previous(self, event):
        """Предыдущее изображение"""
        self.current_index = (self.current_index - 1) % len(self.images)
        self.update_plot()

    def save_current(self, event):
        """Сохранить текущее изображение"""
        filename = f"radar_gallery_{self.current_index:03d}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Изображение сохранено как {filename}")
