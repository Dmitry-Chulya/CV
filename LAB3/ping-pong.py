import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# --- Настройки ---
WIDTH, HEIGHT = 800, 400
BALL_SPEED = 5
PADDLE_SPEED = 10
EPISODES = 1  # количество эпизодов
GAMMA = 0.9  # коэффициент дисконтирования
ALPHA = 0.01  # скорость обучения
EPSILON = 1.0  # начальное значение epsilon
EPSILON_DECAY = 0.995  # уменьшение epsilon
MIN_EPSILON = 0.01  # минимальное значение epsilon

# --- Инициализация Pygame ---
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong with Q-Learning")
clock = pygame.time.Clock()

# --- Нейронная сеть для Q-Learning ---
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 24)  # Вход: 4 состояния (положение мяча и ракеток)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)  # Выход: 3 действия (остановиться, вверх, вниз)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Класс игры Пинг-понг ---
class PingPongGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ball_pos = [WIDTH // 2, HEIGHT // 2]
        self.ball_vel = [BALL_SPEED * random.choice([-1, 1]), BALL_SPEED * random.choice([-1, 1])]
        self.paddle1_pos = HEIGHT // 2 - 50  # Ракетка ИИ
        self.paddle2_pos = HEIGHT // 2 - 50  # Ракетка игрока
        return self.get_state()

    def get_state(self):
        return np.array([self.ball_pos[0], self.ball_pos[1], self.paddle1_pos, self.paddle2_pos], dtype=np.float32)

    def step(self, action1, action2):
        # Движение ракетки ИИ
        if action1 == 1 and self.paddle1_pos > 0:  # Вверх
            self.paddle1_pos -= PADDLE_SPEED
        elif action1 == 2 and self.paddle1_pos < HEIGHT - 100:  # Вниз
            self.paddle1_pos += PADDLE_SPEED

        # Движение ракетки игрока
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.paddle2_pos > 0:  # Вверх
            self.paddle2_pos -= PADDLE_SPEED
        elif keys[pygame.K_DOWN] and self.paddle2_pos < HEIGHT - 100:  # Вниз
            self.paddle2_pos += PADDLE_SPEED

        # Движение мяча
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Проверка на столкновение с верхней и нижней границей
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= HEIGHT:
            self.ball_vel[1] *= -1

        # Проверка на столкновение с ракетками
        if (self.ball_pos[0] <= 20 and self.paddle1_pos <= self.ball_pos[1] <= self.paddle1_pos + 100) or \
           (self.ball_pos[0] >= WIDTH - 20 and self.paddle2_pos <= self.ball_pos[1] <= self.paddle2_pos + 100):
            self.ball_vel[0] *= -1

        # Проверка на выход мяча за границы
        if self.ball_pos[0] < 0:
            return self.reset(), -1, True  # Игрок 2 получает очко
        elif self.ball_pos[0] > WIDTH:
            return self.reset(), 1, True  # Игрок 1 получает очко

        return self.get_state(), 0, False

    def draw(self):
        win.fill((0, 0, 0))
        pygame.draw.rect(win, (255, 255, 255), (10, self.paddle1_pos, 10, 100))  # Ракетка ИИ
        pygame.draw.rect(win, (255, 255, 255), (WIDTH - 20, self.paddle2_pos, 10, 100))  # Ракетка игрока
        pygame.draw.circle(win, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 10)  # Мяч
        pygame.display.update()

# --- Основной цикл обучения ---
def train():
    game = PingPongGame()
    q_network = QNetwork()
    optimizer = optim.Adam(q_network.parameters(), lr=ALPHA)
    criterion = nn.MSELoss()

    global EPSILON

    for episode in range(EPISODES):
        state = game.reset()
        done = False

        while not done:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Преобразование состояния в тензор
            state_tensor = torch.FloatTensor(state)

            # Выбор действия для игрока ИИ
            if random.random() < EPSILON:
                action1 = random.randint(0, 2)  # Случайное действие
            else:
                with torch.no_grad():
                    action1 = torch.argmax(q_network(state_tensor)).item()

            # Действие для игрока (человека) будет определяться в методе step
            action2 = 0  # Игрок управляет ракеткой через клавиши

            next_state, reward, game_over = game.step(action1, action2)

            # Обучение нейронной сети
            next_state_tensor = torch.FloatTensor(next_state)
            target = reward + GAMMA * torch.max(q_network(next_state_tensor)).item()
            target_f = q_network(state_tensor)
            target_f[action1] = target

            optimizer.zero_grad()
            loss = criterion(q_network(state_tensor), target_f)
            loss.backward()
            optimizer.step()

            state = next_state
            game.draw() 
            clock.tick(60)

            if game_over:
                print(f"Эпизод {episode + 1}/{EPISODES} завершен.")

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    pygame.quit()

if __name__ == "__main__":
    train()
