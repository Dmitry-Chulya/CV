# Импорт необходимых библиотек
import numpy as np              # Для работы с числовыми массивами
import torch                    # Основная библиотека PyTorch
import torch.nn as nn          # Модули нейронных сетей
import matplotlib.pyplot as plt # Для визуализации
from sklearn.preprocessing import StandardScaler  # Для нормализации данных

# Гиперпараметры и настройки
SEQ_LENGTH = 50          # Длина входной последовательности
TOTAL_SEQ = 1000        # Общее количество точек в синусоиде
EPOCHS = 40         # Максимальное количество эпох обучения
BATCH_SIZE = 32         # Размер мини-батча
LR = 0.01              # Скорость обучения

# Настройка устройства - использование GPU при наличии
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Генерация синусоидальных данных
x = np.linspace(0, 100, TOTAL_SEQ)  # Создание равномерной сетки точек
data = np.sin(x)                     # Генерация синусоиды

# Нормализация данных
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Функция создания последовательностей
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return torch.tensor(xs).float().unsqueeze(-1), torch.tensor(ys).float().unsqueeze(-1)

# Создание последовательностей и перемещение на выбранное устройство
X, y = create_sequences(data_normalized, SEQ_LENGTH)
X, y = X.to(device), y.to(device)

# Разделение данных на обучающую и тестовую выборки
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Определение модели RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Берём только последний output
        return out

# Инициализация модели, функции потерь и оптимизатора
model = SimpleRNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Обучение модели
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Оценка на тестовой выборке
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().cpu().numpy()  # Перемещение на CPU для визуализации
    y_true = y_test.squeeze().cpu().numpy()

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(preds, label='Предсказание', color='green')
plt.plot(y_true, label='Реальные значения', color='blue')
plt.legend()
plt.title("RNN предсказание синусоиды")
plt.xlabel("Время")
plt.ylabel("Значение")
plt.show()
