import numpy as np

class IntelligentDriverModel:
    """
    Continuous Car-Following Model (IDM) capturing nonlinear driver behavior,
    headway safety gaps, and stop-and-go phantom shockwaves across a ring highway.
    """
    def __init__(self, num_vehicles=60, track_length=2000.0, seed=42):
        np.random.seed(seed)
        self.N = num_vehicles
        self.L = track_length
        
        # Vehicle positions evenly distributed along corridor
        self.positions = np.linspace(0, self.L, self.N, endpoint=False)
        self.velocities = np.full(self.N, 22.0, dtype=np.float64)  # ~80 km/h
        
        # IDM Parameters
        self.v0 = 28.0       # Desired velocity (m/s) ~100 km/h
        self.T_gap = 1.4     # Safe time headway (s)
        self.a_max = 1.8     # Maximum acceleration (m/s^2)
        self.b_comf = 2.0    # Comfortable deceleration (m/s^2)
        self.s0 = 3.5        # Jam distance (m)
        self.delta = 4.0     # Acceleration exponent
        
        # Introduce initial perturbation in lead vehicle to induce shockwave
        self.velocities[0] = 5.0
        
    def step(self, pacing_action=0.0, dt=0.5):
        """
        Advances traffic by dt. pacing_action is an autonomous vehicle control
        speed restriction applied to smooth shockwaves.
        """
        accelerations = np.zeros(self.N)
        
        for i in range(self.N):
            lead_idx = (i + 1) % self.N
            delta_x = (self.positions[lead_idx] - self.positions[i]) % self.L
            delta_v = self.velocities[i] - self.velocities[lead_idx]
            
            # Target velocity adjusted by pacing action
            effective_v0 = max(10.0, self.v0 - pacing_action)
            
            # Dynamic desired gap s*
            s_star = self.s0 + max(0.0, self.velocities[i] * self.T_gap + (self.velocities[i] * delta_v) / (2.0 * np.sqrt(self.a_max * self.b_comf)))
            
            # IDM Acceleration formula
            acc = self.a_max * (1.0 - (self.velocities[i] / effective_v0)**self.delta - (s_star / max(0.1, delta_x))**2)
            
            # Human stochasticity (random slight braking)
            if np.random.rand() < 0.08:
                acc -= 0.6
                
            accelerations[i] = acc
            
        self.velocities = np.clip(self.velocities + accelerations * dt, 0.0, 32.0)
        self.positions = (self.positions + self.velocities * dt) % self.L
        
        return self.positions.copy(), self.velocities.copy()
