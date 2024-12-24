import numpy as np
import random
import matplotlib.pyplot as plt

# Инициализация параметров
GLASS_COUNT = 5  # Количество стаканов
MAX_VOLUME = 10  # Максимальная емкость каждого стакана


# Генерация случайного состояния стаканов
def create_random_state():
    return np.array([random.randint(0, MAX_VOLUME) for _ in range(GLASS_COUNT)])


# Расчет целевого объема жидкости в каждом стакане
def calculate_target_volume(current_state):
    return np.mean(current_state)


# Получение списка возможных действий (переливаний)
def list_possible_moves(current_state):
    moves = []
    for from_idx in range(GLASS_COUNT):
        for to_idx in range(GLASS_COUNT):
            if from_idx != to_idx:
                moves.append((from_idx, to_idx))
    return moves


# Выполнение переливания между стаканами
def perform_transfer(current_state, transfer_action):
    from_glass, to_glass = transfer_action
    transfer_amount = min(current_state[from_glass],
                          MAX_VOLUME - current_state[to_glass],
                          random.randint(1, 5))
    new_state = current_state.copy()
    new_state[from_glass] -= transfer_amount
    new_state[to_glass] += transfer_amount
    return new_state


# Вычисление награды за состояние
def compute_reward(new_state, previous_state):
    goal_volume = calculate_target_volume(new_state)
    total_diff = np.sum(np.abs(new_state - goal_volume))
    previous_diff = np.sum(np.abs(previous_state - goal_volume))

    if total_diff == 0:
        return 10  # Максимальная награда за достижение цели
    elif total_diff < previous_diff:
        return 1  # Награда за приближение к цели
    else:
        return -1  # Штраф за отдаление от цели


# Реализация Q-learning агента
def display_glass_state(state):
    plt.bar(range(len(state)), state, color='orange', alpha=0.7)
    plt.ylim(0, MAX_VOLUME)
    plt.xlabel("Номер стакана")
    plt.ylabel("Объем жидкости")
    plt.title("Текущее состояние стаканов")
    plt.show()


class QLearningAgent:
    def __init__(self, exploration_rate=0.1, learning_rate=0.5, discount_factor=0.9):
        self.exploration_rate = exploration_rate  # Вероятность случайного действия
        self.learning_rate = learning_rate  # Скорость обучения
        self.discount_factor = discount_factor  # Коэффициент дисконтирования
        self.q_values = {}  # Таблица Q-значений
        self.rewards_log = []  # Лог вознаграждений
        self.steps_log = []  # Лог шагов

    # Преобразование состояния в ключ для Q-таблицы
    def state_to_key(self, state):
        return tuple(state)

    # Выбор действия на основе стратегии epsilon-greedy
    def select_action(self, current_state):
        state_key = self.state_to_key(current_state)
        if random.uniform(0, 1) < self.exploration_rate or state_key not in self.q_values:
            return random.choice(list_possible_moves(current_state))
        else:
            actions = list_possible_moves(current_state)
            q_values = [self.q_values[state_key].get(action, 0) for action in actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(actions, q_values) if q == max_q]
            return random.choice(best_actions)

    # Обновление Q-значений
    def update_q_table(self, current_state, action, reward, next_state):
        state_key = self.state_to_key(current_state)
        next_state_key = self.state_to_key(next_state)

        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = {}

        future_best_q = max([self.q_values[next_state_key].get(a, 0) for a in list_possible_moves(next_state)])
        old_q = self.q_values[state_key].get(action, 0)
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * future_best_q - old_q)
        self.q_values[state_key][action] = new_q

    # Основной цикл обучения
    def train_agent(self, total_episodes=1000, max_steps=5000):
        for episode in range(total_episodes):
            current_state = create_random_state()
            cumulative_reward = 0
            steps = 0
            previous_state = current_state.copy()

            for step in range(max_steps):
                action = self.select_action(current_state)
                next_state = perform_transfer(current_state, action)
                reward = compute_reward(next_state, previous_state)
                self.update_q_table(current_state, action, reward, next_state)

                cumulative_reward += reward
                current_state = next_state
                previous_state = current_state.copy()
                steps += 1

                if np.sum(np.abs(current_state - calculate_target_volume(current_state))) == 0:
                    break

            self.exploration_rate = max(0.1, self.exploration_rate * 0.99)
            self.learning_rate = max(0.1, self.learning_rate * 0.99)

            self.rewards_log.append(cumulative_reward)
            self.steps_log.append(steps)

            if episode % 100 == 0:
                print(f"Эпизод {episode + 1}: Общая награда = {cumulative_reward}, Шаги = {steps}")

            if episode % 500 == 0:
                print(f"Состояние стаканов в эпизоде {episode + 1}")
                display_glass_state(current_state)

        self.plot_training_metrics()

    # Построение графиков обучения
    def plot_training_metrics(self):
        def smooth_curve(data, window):
            return np.convolve(data, np.ones(window) / window, mode='valid')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        smoothed_rewards = smooth_curve(self.rewards_log, 50)
        ax1.plot(smoothed_rewards, color='blue', alpha=0.7, label="Вознаграждение")
        ax1.set_xlabel("Эпизоды")
        ax1.set_ylabel("Вознаграждение")
        ax1.set_title("Динамика вознаграждений")

        smoothed_steps = smooth_curve(self.steps_log, 50)
        ax2.plot(smoothed_steps, color='green', alpha=0.7, label="Шаги")
        ax2.set_xlabel("Эпизоды")
        ax2.set_ylabel("Количество шагов")
        ax2.set_title("Динамика количества шагов")

        plt.tight_layout()
        plt.show()

        avg_steps_start = np.mean(self.steps_log[:100])
        avg_steps_end = np.mean(self.steps_log[-100:])

        print(f"Среднее шагов (первые 100): {avg_steps_start}, Среднее шагов (последние 100): {avg_steps_end}")


# Запуск обучения
agent = QLearningAgent(exploration_rate=0.5, learning_rate=0.5)
agent.train_agent(total_episodes=1000)
