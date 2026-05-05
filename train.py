!pip install ultralytics


from ultralytics import YOLO


# Load a model
model = YOLO("yolo11s.pt")


from google.colab import drive
drive.mount('/content/drive')


!ls /content/drive/MyDrive/dataset2


# Tworzymy plik data.yaml
data_yaml = """
path: /content/drive/MyDrive/dataset2
train: images/train
val: images/valid
test: images/test
names:
  0: 'cane'
"""
with open('data.yaml', 'w') as f:
    f.write(data_yaml)
print("Plik data.yaml utworzony!")


from ultralytics import YOLO
# Wczytaj pretrenowany model YOLOv11
model = YOLO("yolo11s.pt")
# Uruchom trening z dostosowanymi parametrami
results = model.train(
    data="data.yaml",         # Ścieżka do pliku YAML
    epochs=5,                # Liczba epok
    batch=16,                 # Zwiększony batch size (przy odpowiednim GPU)
    imgsz=640,                # Rozdzielczość obrazów
    project="runs/train",     # Gdzie zapisać wyniki treningu
    name="cane",     # Nazwa treningu
    lr0=0.001,                # Początkowa wartość learning rate
    patience=5,               # Wczesne zatrzymanie, jeśli nie ma poprawy
    augment=True,             # Zaawansowana augmentacja danych
    workers=2                 # Liczba wątków do przetwarzania danych
)
