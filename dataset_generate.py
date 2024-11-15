from preprocessing import get_titleset, get_backgrounds
import cv2
import random
from dataclasses import dataclass
import asyncio
import numpy as np


@dataclass
class TileImageGenerator:
    background: str
    title: str
    count: int = 0
    limit: int = 0
    title_max_proportion: float = 0.2
    title_min_proportion: float = 0.1

    def __post_init__(self):
        self.background = cv2.imread(self.background)
        self.title = cv2.imread(self.title)

    def __iter__(self):
        pass

    def __next__(self):
        if self.limit == 0 or self.count < self.limit:
            pass
        else:
            raise StopAsyncIteration
        self.count += 1

    def __len__(self):
        return self.count



    def _process_image(self) -> cv2.Mat:
        # Set a random title size as a proportion of the background
        background_height, background_weight = self.background.shape[:2]
        title_height, title_weight = self.title.shape[:2]

        # Изменяем размер тайла
        new_title_height = random.choice(range(
            int(background_height * self.title_min_proportion),
            int(background_height * self.title_max_proportion)
        ))
        aspect_ratio = title_weight / title_height
        new_title_weight = int(new_title_height * aspect_ratio)
        resized_title = cv2.resize(self.title, (new_title_weight, new_title_height))

        # Set a random angle for the title
        angle = random.randint(0, 360)
        center = (new_title_weight / 2, new_title_height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # Рассчитываем новый размер холста для поворота
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        bound_w = int(new_title_height * sin + new_title_weight * cos)
        bound_h = int(new_title_height * cos + new_title_weight * sin)

        # Корректируем матрицу поворота для нового холста
        rotation_matrix[0, 2] += bound_w / 2 - new_title_weight / 2
        rotation_matrix[1, 2] += bound_h / 2 - new_title_height / 2

        # Поворачиваем тайл с новым холстом
        rotated_title = cv2.warpAffine(
            resized_title, rotation_matrix, (bound_w, bound_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)  # Установка прозрачного фона
        )

        # Создаем маску для наложения тайла
        rotated_gray = cv2.cvtColor(rotated_title, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(rotated_gray, 1, 255, cv2.THRESH_BINARY)

        # Создаем копию фона
        background_copy = self.background.copy()

        # Устанавливаем случайную позицию для тайла на фоне
        x = random.randint(0, background_weight - bound_w)
        y = random.randint(0, background_height - bound_h)

        # Выделяем область на фоне и используем маску для наложения
        roi = background_copy[y:y + bound_h, x:x + bound_w]
        roi[np.where(mask)] = rotated_title[np.where(mask)]

        # Помещаем измененный участок обратно на фон
        background_copy[y:y + bound_h, x:x + bound_w] = roi

        return background_copy


if __name__ == '__main__':
    backgrounds: list = get_backgrounds()
    first_generator = TileImageGenerator(background=backgrounds[1], title=get_titleset()[0][0], count=0, limit=10)
    cv2.imshow('background', first_generator._process_image())
    cv2.waitKey(0)
    cv2.destroyAllWindows()