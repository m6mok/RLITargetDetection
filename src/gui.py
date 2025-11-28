import threading
from typing import TYPE_CHECKING
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from src.utils.io import read_text

if TYPE_CHECKING:
    from typing import Callable

    from src.detector import RADARDetector
    from src.data_generator import RADARDataGenerator
    from src.visualizer import RADARVisualizer

    DetectorFabric = Callable[[], "RADARDetector"] | None
    DataGeneratorFabric = Callable[[], "RADARDataGenerator"] | None
    VisualizerFabric = Callable[[], "RADARVisualizer"] | None

INFO_TEXT_PATH = Path("data/info_text.txt")


class RADARGUI:
    """Графический интерфейс для детекции целей на РЛИ"""

    def __init__(
        self,
        detector_fabric: "DetectorFabric" = None,
        data_generator_fabric: "DataGeneratorFabric" = None,
        visualizer_fabric: "VisualizerFabric" = None,
    ) -> None:
        assert detector_fabric is not None, "Detector is required"
        assert data_generator_fabric is not None, "Data generator is required"

        self.root = tk.Tk()
        self.root.title("Детекция целей на радиолокационных изображениях")
        self.root.geometry("800x600")

        # Инициализация детектора и генератора
        self.detector = detector_fabric()
        self.generator = data_generator_fabric()
        self.visualizer_fabric = visualizer_fabric

        self.metrics: dict | None = None

        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Создание вкладок
        notebook = ttk.Notebook(self.root)

        # Вкладка основной информации
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Главная")

        # Вкладка генерации данных
        generation_frame = ttk.Frame(notebook)
        notebook.add(generation_frame, text="Генерация данных")

        # Вкладка детекции
        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text="Детекция")

        # Вкладка оценки
        evaluation_frame = ttk.Frame(notebook)
        notebook.add(evaluation_frame, text="Оценка модели")

        notebook.pack(expand=True, fill="both")

        # Заполнение вкладок
        self.setup_main_tab(main_frame)
        self.setup_generation_tab(generation_frame)
        self.setup_detection_tab(detection_frame)
        self.setup_evaluation_tab(evaluation_frame)

    def setup_main_tab(self, parent):
        """Настройка главной вкладки"""
        title_label = tk.Label(
            parent,
            text="Детекция целей на радиолокационных изображениях",
            font=("Arial", 16, "bold"),
        )
        title_label.pack(pady=20)

        info_text = read_text(INFO_TEXT_PATH)
        if info_text is None:
            raise RuntimeError(f"No text in path: {INFO_TEXT_PATH}")

        info_label = tk.Label(
            parent, text=info_text, justify=tk.LEFT, font=("Arial", 11)
        )
        info_label.pack(pady=20, padx=20)

        status_frame = tk.Frame(parent)
        status_frame.pack(pady=10)

        device_label = tk.Label(
            status_frame,
            text=f"Устройство: {self.detector.device}",
            font=("Arial", 10)
        )
        device_label.pack()

        model_label = tk.Label(
            status_frame,
            text="Модель: YOLOv8",
            font=("Arial", 10)
        )
        model_label.pack()

    def setup_generation_tab(self, parent):
        """Настройка вкладки генерации данных"""
        # Генерация одиночного примера
        single_frame = ttk.LabelFrame(parent, text="Генерация одиночного РЛИ")
        single_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            single_frame,
            text="Сгенерировать и показать пример",
            command=self.generate_single_sample,
        ).pack(pady=5)

        # Генерация датасета
        dataset_frame = ttk.LabelFrame(parent, text="Генерация датасета")
        dataset_frame.pack(fill="x", padx=10, pady=5)

        dataset_subframe = tk.Frame(dataset_frame)
        dataset_subframe.pack(pady=5)

        tk.Label(dataset_subframe, text="Количество образцов:").grid(
            row=0, column=0, padx=5
        )
        self.samples_var = tk.StringVar(value="100")
        samples_entry = tk.Entry(
            dataset_subframe, textvariable=self.samples_var, width=10
        )
        samples_entry.grid(row=0, column=1, padx=5)

        tk.Label(dataset_subframe, text="Путь сохранения:").grid(
            row=0, column=2, padx=5
        )
        self.save_path_var = tk.StringVar(value="radar_dataset")
        save_entry = tk.Entry(
            dataset_subframe, textvariable=self.save_path_var, width=20
        )
        save_entry.grid(row=0, column=3, padx=5)

        tk.Button(
            dataset_subframe,
            text="Обзор",
            command=self.browse_save_path
        ).grid(
            row=0, column=4, padx=5
        )

        tk.Button(
            dataset_frame,
            text="Сгенерировать датасет",
            command=self.generate_dataset
        ).pack(pady=5)

        self.gen_status = tk.Label(parent, text="", fg="blue")
        self.gen_status.pack()

    def setup_detection_tab(self, parent):
        """Настройка вкладки детекции"""
        # Обработка пользовательского изображения
        user_frame = ttk.LabelFrame(
            parent, text="Обработка пользовательского изображения"
        )
        user_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            user_frame,
            text="Выбрать изображение",
            command=self.process_user_image
        ).pack(pady=5)

        self.user_image_path = tk.Label(
            user_frame, text="Файл не выбран", wraplength=500
        )
        self.user_image_path.pack(pady=5)

        # Детекция на синтетических данных
        synthetic_frame = ttk.LabelFrame(
            parent, text="Детекция на синтетических данных"
        )
        synthetic_frame.pack(fill="x", padx=10, pady=5)

        synth_subframe = tk.Frame(synthetic_frame)
        synth_subframe.pack(pady=5)

        tk.Label(synth_subframe, text="Количество примеров:").grid(
            row=0, column=0, padx=5
        )
        self.synth_samples_var = tk.StringVar(value="50")
        synth_entry = tk.Entry(
            synth_subframe, textvariable=self.synth_samples_var, width=10
        )
        synth_entry.grid(row=0, column=1, padx=5)

        tk.Button(
            synthetic_frame,
            text="Запустить детекцию",
            command=self.run_synthetic_detection,
        ).pack(pady=5)

        self.detection_status = tk.Label(parent, text="", fg="blue")
        self.detection_status.pack()

    def setup_evaluation_tab(self, parent):
        """Настройка вкладки оценки модели"""
        # Оценка на сохраненном датасете
        eval_frame = ttk.LabelFrame(parent, text="Оценка на датасете")
        eval_frame.pack(fill="x", padx=10, pady=5)

        eval_subframe = tk.Frame(eval_frame)
        eval_subframe.pack(pady=5)

        tk.Label(eval_subframe, text="Путь к датасету:").grid(
            row=0,
            column=0,
            padx=5
        )
        self.dataset_path_var = tk.StringVar(value="radar_dataset")
        dataset_entry = tk.Entry(
            eval_subframe, textvariable=self.dataset_path_var, width=20
        )
        dataset_entry.grid(row=0, column=1, padx=5)

        tk.Button(
            eval_subframe,
            text="Обзор",
            command=self.browse_dataset_path
        ).grid(
            row=0, column=2, padx=5
        )

        tk.Button(
            eval_frame, text="Оценить модель", command=self.evaluate_on_dataset
        ).pack(pady=5)

        # Визуализация статистики
        stats_frame = ttk.LabelFrame(parent, text="Визуализация статистики")
        stats_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            stats_frame,
            text="Показать статистику",
            command=self.show_statistics,
            state=tk.DISABLED,
        ).pack(pady=5)
        self.stats_button = stats_frame.winfo_children()[-1]

        # Поле для вывода результатов
        self.results_text = scrolledtext.ScrolledText(
            parent,
            height=15,
            width=70
        )
        self.results_text.pack(padx=10, pady=10, fill="both", expand=True)

        self.eval_status = tk.Label(parent, text="", fg="blue")
        self.eval_status.pack()

    def browse_save_path(self):
        """Выбор пути для сохранения"""
        if len(path := filedialog.askdirectory()) > 0:
            self.save_path_var.set(path)

    def browse_dataset_path(self):
        """Выбор пути к датасету"""
        if len(path := filedialog.askdirectory()) > 0:
            self.dataset_path_var.set(path)

    def generate_single_sample(self):
        """Генерация одиночного примера"""
        try:
            image, bboxes, params = self.generator.generate_sample()

            # Визуализация
            visualizer = self.visualizer_fabric()
            visualizer.visualize_detection(
                image, bboxes, bboxes, params, "Сгенерированное РЛИ"
            )

            self.update_status(
                self.gen_status,
                "Пример успешно сгенерирован и показан"
            )
        except Exception as e:
            error_message = str(e)
            messagebox.showerror(
                "Ошибка",
                f"Ошибка при генерации: {error_message}",
            )

    def generate_dataset(self):
        """Генерация датасета"""

        def generate_thread():
            try:
                num_samples = int(self.samples_var.get())
                save_path = self.save_path_var.get()

                self.update_status(self.gen_status, "Генерация датасета...")

                # Генерация в отдельном потоке
                result_path = self.generator.generate_dataset(
                    num_samples,
                    save_path
                )

                self.root.after(
                    0,
                    lambda: self.update_status(
                        self.gen_status,
                        f"Датасет успешно сгенерирован: {result_path}"
                    ),
                )

            except Exception as e:
                error_message = str(e)
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Ошибка",
                        f"Ошибка при генерации датасета: {error_message}"
                    ),
                )

        # Запуск в отдельном потоке
        threading.Thread(target=generate_thread, daemon=True).start()

    def process_user_image(self):
        """Обработка пользовательского изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите РЛИ",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )

        if file_path:
            self.user_image_path.config(text=file_path)

            try:
                self.detector.process_user_image(file_path)
                self.update_status(
                    self.detection_status,
                    "Обработка завершена"
                )
            except Exception as e:
                error_message = str(e)
                messagebox.showerror(
                    "Ошибка",
                    f"Ошибка при обработке изображения: {error_message}"
                )

    def _synthetic_detection(self) -> None:
        try:
            num_samples = int(self.synth_samples_var.get())

            self.update_status(
                self.detection_status,
                "Запуск детекции на синтетических данных..."
            )

            # Запуск детекции
            self.metrics = self.detector.evaluate_on_synthetic_data(
                num_samples, use_gallery=True
            )
            self.root.after(0, self.show_detection_results)

        except Exception as e:
            error_message = str(e)
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Ошибка",
                    f"Ошибка при детекции: {error_message}"
                ),
            )

    def run_synthetic_detection(self):
        """Запуск детекции на синтетических данных"""
        thread = threading.Thread(
            target=self._synthetic_detection,
            daemon=True
        )
        thread.start()

    def _evaluation_on_dataset(self):
        try:
            dataset_path = self.dataset_path_var.get()

            self.update_status(
                self.eval_status,
                "Оценка модели на датасете...",
            )

            # Запуск оценки
            self.metrics = self.detector.evaluate_on_saved_dataset(
                dataset_path
            )
            self.root.after(0, self.show_evaluation_results)

        except Exception as e:
            error_message = str(e)
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Ошибка", f"Ошибка при оценке: {error_message}"
                ),
            )

    def evaluate_on_dataset(self) -> None:
        """Оценка модели на датасете"""
        thread = threading.Thread(
            target=self._evaluation_on_dataset,
            daemon=True
        )
        thread.start()

    def show_results(self, message: str) -> None:
        assert isinstance(self.metrics, dict)

        results_text = "\n".join((
            "=== РЕЗУЛЬТАТЫ ОЦЕНКИ ===\n",
            f"Precision: {self.metrics.get('precision', 0):.3f}"
            f"Recall: {self.metrics.get('recall', 0):.3f}"
            f"F1-Score: {self.metrics.get('f1_score', 0):.3f}"
            f"mAP@0.5: {self.metrics.get('map50', 0):.3f}"
        ))

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)

        # Активируем кнопку показа статистики
        self.stats_button.config(state=tk.NORMAL)

        self.update_status(self.detection_status, message)

    def show_detection_results(self) -> None:
        """Показать результаты детекции"""
        self.show_results("Детекция завершена")

    def show_evaluation_results(self) -> None:
        """Показать результаты оценки"""
        self.show_results("Оценка завершена")

    def show_statistics(self) -> None:
        """Показать статистику"""
        assert isinstance(self.metrics, dict)
        self.detector.visualizer.plot_statistics(self.metrics)

    def update_status(self, label: tk.Label, message: str) -> None:
        """Обновление статусного сообщения"""
        label.config(text=message)
        self.root.update_idletasks()

    def run(self) -> None:
        """Запуск приложения"""
        self.root.mainloop()
