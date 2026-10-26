import numpy as np

class TrafficDQNController:
    """
    Deep Q-Network for Autonomous Speed Harmonization and Shockwave Dissipation.
    Learns discrete speed pacing offsets to minimize velocity variance across the corridor.
    """
    def __init__(self, state_dim=4, action_dim=5, hidden_dim=24, lr=0.015):
        np.random.seed(42)
        self.state_dim = state_dim
        self.action_dim = action_dim  # Pacing offsets: [0, 2, 4, 6, 8 m/s]
        self.action_values = [0.0, 2.0, 4.0, 6.0, 8.0]
        
        # Q-Network weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, action_dim))
        
        self.lr = lr
        self.gamma = 0.92
        self.epsilon = 0.25
        self.loss_history = []
        
    def _relu(self, x):
        return np.maximum(0, x)
        
    def forward(self, state):
        state = state.reshape(1, -1)
        self.h = self._relu(np.dot(state, self.W1) + self.b1)
        q_vals = np.dot(self.h, self.W2) + self.b2
        return q_vals.flatten()
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            q_vals = self.forward(state)
            action_idx = np.argmax(q_vals)
        return action_idx, self.action_values[action_idx]
        
    def train_step(self, s, a_idx, r, s_next):
        s = s.reshape(1, -1)
        s_next = s_next.reshape(1, -1)
        
        # Target Q
        next_q = self.forward(s_next)
        target = r + self.gamma * np.max(next_q)
        
        # Current Q
        current_q = self.forward(s)
        td_error = target - current_q[a_idx]
        
        # Backprop through active action branch
        grad_out = np.zeros((1, self.action_dim))
        grad_out[0, a_idx] = - 2.0 * td_error
        
        dW2 = np.dot(self.h.T, grad_out)
        db2 = grad_out
        
        dh = np.dot(grad_out, self.W2.T)
        dz1 = dh * (self.h > 0).astype(float)
        dW1 = np.dot(s.T, dz1)
        db1 = dz1
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        self.loss_history.append(float(td_error**2))
