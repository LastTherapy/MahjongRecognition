import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'titles')
tiles_types = os.listdir(root_dir)

class_labels: dict = {}  # ID -> имя класса
class_images: dict = {}  # ID -> изображения
class_id: int = 0

# Create classes  from directory structure
for tiles_type in tiles_types:
    path_tiles_type: str = os.path.join(root_dir, tiles_type)
    for title in sorted(os.listdir(path_tiles_type)):
        path_titles: str = os.path.join(path_tiles_type, title)
        images_path: list = [os.path.join(path_titles, image)
                             for image in os.listdir(path_titles)
                             if image.endswith(('.png', '.jpg', '.jpeg'))
                             ]
        if len(images_path) > 0:
            class_name: str = f'{tiles_type}_{title}'
            class_labels[class_id] = class_name
            class_images[class_id] = tuple(images_path)
            class_id += 1
#
# print(class_images)
# print(class_labels)


def get_backgrounds() -> list:
    background_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'backgrounds')
    return [os.path.join(background_dir, image) for image in os.listdir(background_dir) if image.endswith(('.png', '.jpg', '.jpeg'))]

def get_titleset() -> dict:
    return class_images

def create_yaml(name: str = 'data.yaml'):
    """
    Create Ultralytics YOLOv8 configuration file (data.yaml) for training.
    Args:
        name (str): Optional name for the file.
    Returns:
        str: Path to the created configuration file.
    """
    with open(name, 'w') as f:
        f.write(f"""path: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')} 
train: images/train
val: images/val
nc: {len(class_labels)}
names: {[label for label in class_labels.values()]}
""")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


if __name__ == '__main__':
    print(create_yaml())