from src.gui import RADARGUI
from src.detector import RADARDetector
from src.data_generator import RADARDataGenerator
from src.visualizer import RADARVisualizer
from src.gallery import RADARGallery


def data_generator_fabric(*args, **kwargs) -> RADARDataGenerator:
    return RADARDataGenerator(*args, **kwargs)


def visualizer_fabric(*args, **kwargs) -> RADARVisualizer:
    return RADARVisualizer(*args, **kwargs)


def gallery_fabric(*args, **kwargs) -> RADARGallery:
    return RADARGallery(
        *args,
        **kwargs,
        visualizer_fabric=visualizer_fabric,
    )


def detector_fabric(*args, **kwargs) -> RADARDetector:
    return RADARDetector(
        *args,
        **kwargs,
        data_generator_fabric=data_generator_fabric,
        visualizer_fabric=visualizer_fabric,
        gallery_fabric=gallery_fabric,
    )


def main():
    """Основная функция программы"""
    print("=== Детекция целей на радиолокационных изображениях ===")

    # Запуск GUI
    app = RADARGUI(
        detector_fabric=detector_fabric,
        data_generator_fabric=data_generator_fabric,
        visualizer_fabric=visualizer_fabric,
    )
    app.run()


if __name__ == "__main__":
    main()
