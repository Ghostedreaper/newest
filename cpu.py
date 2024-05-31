import psutil
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import logging
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from threadpoolctl import threadpool_limits

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Handle OpenMP libraries warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = "/tmp"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define fuzzy variables
risk = ctrl.Antecedent(np.arange(0, 11, 1), 'risk')
uncertainty = ctrl.Antecedent(np.arange(0, 11, 1), 'uncertainty')
decision = ctrl.Consequent(np.arange(0, 11, 1), 'decision')

# Define fuzzy sets
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 5])
risk['medium'] = fuzz.trimf(risk.universe, [0, 5, 10])
risk['high'] = fuzz.trimf(risk.universe, [5, 10, 10])

uncertainty['low'] = fuzz.trimf(uncertainty.universe, [0, 0, 5])
uncertainty['medium'] = fuzz.trimf(uncertainty.universe, [0, 5, 10])
uncertainty['high'] = fuzz.trimf(uncertainty.universe, [5, 10, 10])

decision['low'] = fuzz.trimf(decision.universe, [0, 0, 5])
decision['medium'] = fuzz.trimf(decision.universe, [0, 5, 10])
decision['high'] = fuzz.trimf(decision.universe, [5, 10, 10])

# Define fuzzy rules
rules = [
    ctrl.Rule(risk['low'] & uncertainty['low'], decision['high']),
    ctrl.Rule(risk['medium'] & uncertainty['low'], decision['medium']),
    ctrl.Rule(risk['high'] & uncertainty['low'], decision['low']),
    ctrl.Rule(risk['low'] & uncertainty['medium'], decision['medium']),
    ctrl.Rule(risk['medium'] & uncertainty['medium'], decision['medium']),
    ctrl.Rule(risk['high'] & uncertainty['medium'], decision['low']),
    ctrl.Rule(risk['low'] & uncertainty['high'], decision['low']),
    ctrl.Rule(risk['medium'] & uncertainty['high'], decision['low']),
    ctrl.Rule(risk['high'] & uncertainty['high'], decision['low']),
]

# Create the fuzzy control system
decision_ctrl = ctrl.ControlSystem(rules)
decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)

# Define the model
class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def gather_system_process_data():
    data = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            info = proc.info
            io = proc.io_counters() if hasattr(proc, 'io_counters') else (0, 0, 0, 0)
            data.append([
                info['cpu_percent'],
                info['memory_percent'],
                io[0],  # read_count
                io[1],  # write_count
                io[2],  # read_bytes
                io[3],  # write_bytes
            ])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return np.array(data)

def run_pytorch_example():
    data = gather_system_process_data()
    if data.size == 0:
        logging.warning("No data collected from system processes.")
        return

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    labels = (data[:, 0] > np.median(data[:, 0])).astype(int)  # Example: label based on CPU usage

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Ensure that all elements of y_train and y_test are between 0 and 1
    y_train = y_train.clamp(0, 1)
    y_test = y_test.clamp(0, 1)


    # Define the model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    ).to(device)

    # Train the model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 10
    batch_size = 32

    model.train()
    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    # Save the model
    model_save_path = 'classification_model.pth'
    torch.save(model.state_dict(), model_save_path)
    logging.info(f'Saved model to {model_save_path}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device))
        accuracy = (y_pred.round() == y_test.to(device)).float().mean()
        logging.info(f'Test Accuracy: {accuracy:.4f}')

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=128, model_save_path='dqn_model.pth'):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=200000)
        self.gamma = 0.97  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.000001
        self.model_save_path = model_save_path
        self.model, self.optimizer = self.build_model()
        self.target_model, _ = self.build_model()
        self.load_model()
        self.update_target_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model.to(device), optimizer

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            target = self.model(state)
            if done:
                target[0][action] = reward
            else:
                with torch.no_grad():
                    t = self.target_model(next_state)
                target[0][action] = reward + self.gamma * torch.max(t)
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self):
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=device))
            logging.info("Loaded model from {}".format(self.model_save_path))
        else:
            logging.info("No existing model found, training from scratch.")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        logging.info("Saved model to {}".format(self.model_save_path))

def adjust_openmp_settings(cpu_count):
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    logging.info(f"Set OMP_NUM_THREADS to {cpu_count}")

def synthetic_benchmark(num_runs=5):
    total_time = 0.0
    for _ in range(num_runs):
        start_time = time.time()
        [x**2 for x in range(10000)]
        end_time = time.time()
        total_time += (end_time - start_time)
    avg_time = total_time / num_runs
    return avg_time

def cpu_load_balancer(agent, state_size, interval=30):
    while True:
        avg_benchmark_time = synthetic_benchmark()
        logging.info(f"Average synthetic benchmark time: {avg_benchmark_time:.4f} seconds")
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
        max_cpu = max(cpu_percentages)
        min_cpu = min(cpu_percentages)
        imbalance = max_cpu - min_cpu

        # Apply fuzzy logic to decide on adjustments
        decision_sim.input['risk'] = imbalance
        decision_sim.input['uncertainty'] = avg_benchmark_time
        decision_sim.compute()
        decision_level = decision_sim.output['decision']
        logging.info(f"Fuzzy decision level: {decision_level}")

        state = [mp.cpu_count(), avg_benchmark_time, max_cpu, min_cpu, imbalance, decision_level]
        action = agent.act(state)

        actions = [
            (mp.cpu_count(), 1),  # Example: no change
            (mp.cpu_count(), 2),  # Example: modify KMP_BLOCKTIME
            (mp.cpu_count() - 1, 1),  # Example: reduce CPU count
            (mp.cpu_count() + 1, 1),  # Example: increase CPU count
        ]

        cpu_count, kmp_blocktime = actions[action]

        adjust_openmp_settings(cpu_count)
        os.environ["KMP_BLOCKTIME"] = str(kmp_blocktime)

        reward = -avg_benchmark_time  # Negative because we want to minimize time
        next_state = [mp.cpu_count(), avg_benchmark_time, max_cpu, min_cpu, imbalance, decision_level]
        done = False

        agent.remember(state, action, reward, next_state, done)
        agent.replay(agent.batch_size)
        agent.update_target_model()
        agent.save_model()

        time.sleep(interval)

def redistribute_load():
    num_processes = mp.cpu_count()
    processes = []

    for _ in range(num_processes):
        p = mp.Process(target=busy_work)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def busy_work():
    start_time = time.time()
    while time.time() - start_time < 1:
        pass

def monitor_and_adjust_resources(cpu_threshold=70, ram_threshold=70, sleep_interval=5):
    logging.info("Starting resource manager.")
    while True:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                cpu_usage = proc.info['cpu_percent']
                ram_usage = proc.info['memory_percent']

                if cpu_usage is None or ram_usage is None:
                    continue

                if cpu_usage > cpu_threshold or ram_usage > ram_threshold:
                    logging.info(f"Process {proc.info['name']} (PID: {proc.info['pid']}) is using {cpu_usage}% CPU and {ram_usage}% RAM.")
                    process = psutil.Process(proc.info['pid'])

                    if cpu_usage > cpu_threshold:
                        try:
                            process.nice(psutil.IDLE_PRIORITY_CLASS)
                            logging.info(f"Reduced CPU priority of process {proc.info['name']} (PID: {proc.info['pid']}).")
                        except AttributeError:
                            process.nice(1)
                            logging.info(f"Reduced CPU priority of process {proc.info['name']} (PID: {proc.info['pid']}) using Unix nice value.")
                    if ram_usage > ram_threshold:
                        process.terminate()
                        logging.info(f"Terminated process {proc.info['name']} (PID: {proc.info['pid']}) due to high RAM usage.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        time.sleep(sleep_interval)

if __name__ == "__main__":
    try:
        # Use threadpoolctl to limit OpenMP library usage
        with threadpool_limits(limits=1, user_api='blas'):
            resource_manager_process = mp.Process(target=monitor_and_adjust_resources, args=(70, 70, 5))
            resource_manager_process.start()

            state_size = 6  # Adjust based on gathered data features
            action_size = 4
            batch_size = 64
            agent = DQNAgent(state_size, action_size, batch_size)
            cpu_load_balancer_process = mp.Process(target=cpu_load_balancer, args=(agent, state_size))
            cpu_load_balancer_process.start()

            run_pytorch_example()
    except KeyboardInterrupt:
        logging.info("Stopping resource manager and CPU load balancer.")
        resource_manager_process.terminate()
        cpu_load_balancer_process.terminate()
