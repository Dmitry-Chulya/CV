import torch  
import torch.nn as nn 
import torch.optim as optim  
from torchvision import datasets, transforms  
import matplotlib.pyplot as plt  

# Проверка на наличие GPU и установка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определение трансформаций для данных (нормализация и преобразование в тензор)
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка обучающего и тестового наборов данных MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Создание загрузчиков данных для батчевой обработки
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Определение архитектуры сверточной нейронной сети (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # Инициализация базового класса
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Первый сверточный слой: 1 входной канал, 32 фильтра, размер ядра 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Второй сверточный слой: 32 входных канала, 64 фильтра
        self.dropout1 = nn.Dropout(0.25)  # Слой Dropout для регуляризации
        self.fc1 = nn.Linear(9216, 128)  # Полносвязный слой: 9216 входов (после свертки и пулинга), 128 выходов
        self.fc2 = nn.Linear(128, 10)  # Полносвязный слой: 128 входов, 10 выходов (для 10 классов)

    def forward(self, x):
        x = self.conv1(x)  # Применяем первый сверточный слой
        x = nn.ReLU()(x)  # Применяем функцию активации ReLU
        x = self.conv2(x) 
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)  # Применяем слой максимального пулинга с размером 2
        x = self.dropout1(x)  # Применяем Dropout для регуляризации
        x = torch.flatten(x, 1)  # Преобразуем тензор в одномерный для полносвязного слоя
        x = self.fc1(x)  # Применяем первый полносвязный слой
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)  # Применяем LogSoftmax для получения вероятностей классов

# Инициализация модели, оптимизатора и функции потерь
model = CNN().to(device)  # Перемещаем модель на выбранное устройство (GPU или CPU)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()  # Используем отрицательную логарифмическую функцию потерь

# Обучение модели
for epoch in range(1, 2): 
    model.train()
    total_loss = 0
    for data, target in train_loader:  # Проходим по батчам обучающего набора
        data, target = data.to(device), target.to(device)  # Перемещаем данные и цели на устройство
        optimizer.zero_grad()  # Сбрасываем градиенты
        output = model(data)  
        loss = criterion(output, target) 
        loss.backward()  # Выполняем обратное распространение ошибки
        optimizer.step()  
        total_loss += loss.item()  
    print(f"Epochs {epoch}, Loss: {total_loss:.4f}")  

# Тестирование модели
model.eval()  # Устанавливаем модель в режим оценки
correct = 0  
with torch.no_grad():  # Отключаем градиенты для тестирования
    for data, target in test_loader: 
        data, target = data.to(device), target.to(device) 
        output = model(data) 
        pred = output.argmax(dim=1) 
        correct += pred.eq(target).sum().item() 
print(f"Test accuracy: {100. * correct / len(test_loader.dataset):.2f}%") 

# Отображение предсказаний
examples = enumerate(test_loader)  
batch_idx, (example_data, example_targets) = next(examples) 

with torch.no_grad():  
    output = model(example_data.to(device)) 

# Визуализация предсказаний
plt.figure(figsize=(10, 3))  # Устанавливаем размер графика
for i in range(6):  # Отображаем 6 примеров
    plt.subplot(1, 6, i + 1)  # Создаем подграфик
    plt.tight_layout()  
    plt.imshow(example_data[i][0], cmap='BuGn', interpolation='none')
    pred_label = output[i].argmax().item()  
    plt.title(f"Predict: {pred_label}")  
    plt.xticks([]), plt.yticks([])  
plt.show()  
