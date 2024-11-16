import cv2
import os
import re
from datetime import datetime
from pathlib import Path
# https://chatgpt.com/c/6724d5fa-2f78-8009-8f55-83a426c8f217?model=o1-mini

# Ruta del directorio de imágenes
image_dir = Path(__file__).parent / "videos" / "escena5"
fps = 5 # Ajustable entre 5 y 10
print(str(image_dir))

# Expresión regular para extraer el formato HH-MM-SS del nombre de archivo
time_pattern = re.compile(r"(\d{2}-\d{2}-\d{2})")

# Leer las imágenes y extraer los tiempos
images_with_times = []
for filename in sorted(os.listdir(image_dir)):
    match = time_pattern.search(filename)
    if match:
        time_str = match.group(1)
        try:
            time_obj = datetime.strptime(time_str, "%H-%M-%S").time()
            image_path = os.path.join(image_dir, filename)
            images_with_times.append((time_obj, image_path))
        except ValueError:
            continue  # Ignora archivos con formatos de tiempo incorrectos

# Ordena las imágenes por hora
images_with_times.sort(key=lambda x: x[0])

# Mostrar las imágenes como un video
for _, image_path in images_with_times:
    img = cv2.imread(image_path)
    if img is None:
        continue  # Ignora archivos que no se pueden cargar

    cv2.imshow("Video", img)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
import cv2
import os
import re
from datetime import datetime

# Ruta del directorio de imágenes
image_dir = "ruta/a/tu/directorio"
fps = 5  # Ajustable entre 5 y 10

# Expresión regular para extraer el formato HH-MM-SS del nombre de archivo
time_pattern = re.compile(r"(\d{2}-\d{2}-\d{2})")

# Leer las imágenes y extraer los tiempos
images_with_times = []
for filename in sorted(os.listdir(image_dir)):
    match = time_pattern.search(filename)
    if match:
        time_str = match.group(1)
        try:
            time_obj = datetime.strptime(time_str, "%H-%M-%S").time()
            image_path = os.path.join(image_dir, filename)
            images_with_times.append((time_obj, image_path))
        except ValueError:
            continue  # Ignora archivos con formatos de tiempo incorrectos

# Ordena las imágenes por hora
images_with_times.sort(key=lambda x: x[0])

# Mostrar las imágenes como un video
for _, image_path in images_with_times:
    img = cv2.imread(image_path)
    if img is None:
        continue  # Ignora archivos que no se pueden cargar

    cv2.imshow("Video", img)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
