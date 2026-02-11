"""
UAV-Assisted VEC Environment
Implements the complete system model from the paper
"""

import numpy as np
import gym
from gym import spaces


class UAVAssistedVECEnv(gym.Env):
    """
    Environment for UAV-Assisted Vehicular Edge Computing
    Implements equations and models from the paper
    """
    
    def __init__(self, config):
        super(UAVAssistedVECEnv, self).__init__()
        
        self.config = config
        self.td3_cfg = config['td3']
        self.net_cfg = config['network']
        self.map_cfg = config['map']
        self.num_vehicles = config['num_vehicles']
        
        # Extract key parameters
        self.tau = self.net_cfg['time_slot_duration']
        self.num_rsus = self.map_cfg['num_rsus']
        self.num_ens = self.num_rsus + 1  # RSUs + UAV
        
        # Initialize RSU positions
        self.rsu_positions = np.array(self.map_cfg['rsu_positions'])
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Initialize state variables
        self.reset()
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space: [v_u, theta_u, alpha_1, ..., alpha_N]
        # v_u ∈ [0, v_max], theta_u ∈ [0, 2π), alpha_n ∈ [0, 1]
        action_low = np.array([0, 0] + [0] * self.num_vehicles)
        action_high = np.array([
            self.net_cfg['uav_max_velocity'],
            2 * np.pi
        ] + [1] * self.num_vehicles)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # State space dimension calculation (from equation 29)
        # W(t): 2(N+1) - coordinates of vehicles and UAV
        # U(t): 2N - task information (D_n, T_max_n)
        # Y(t): N - time intervals
        # Q(t): N + M + 1 - queue backlogs
        # H(t): N(M+1) - channel gains
        # E_res_u: 1 - residual UAV energy
        state_dim = (
            2 * (self.num_vehicles + 1) +  # W(t)
            2 * self.num_vehicles +          # U(t)
            self.num_vehicles +              # Y(t)
            self.num_vehicles + self.num_ens +  # Q(t)
            self.num_vehicles * self.num_ens +  # H(t)
            1                                # E_res_u
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize vehicle positions randomly on the map
        area_width, area_height = self.map_cfg['area']
        self.vehicle_positions = np.random.uniform(
            low=[0, 0],
            high=[area_width, area_height],
            size=(self.num_vehicles, 2)
        )
        
        # Initialize vehicle velocities
        v_min, v_max = self.net_cfg['vehicle_velocity_range']
        self.vehicle_velocities = np.random.uniform(v_min, v_max, self.num_vehicles)
        self.vehicle_angles = np.random.uniform(0, 2*np.pi, self.num_vehicles)
        
        # Initialize UAV position (random or center)
        self.uav_position = np.array([area_width/2, area_height/2])
        self.uav_height = self.net_cfg['uav_height']
        
        # Initialize UAV residual energy
        self.uav_residual_energy = self.net_cfg['uav_max_energy']
        
        # Initialize queue backlogs
        self.vehicle_queues = np.zeros(self.num_vehicles)
        self.en_queues = np.zeros(self.num_ens)
        
        # Initialize task generation time intervals
        self.task_intervals = np.zeros(self.num_vehicles)
        
        # Initialize current tasks
        self.current_tasks = self._generate_tasks()
        
        # Initialize AoI
        self.aoi = np.zeros(self.num_vehicles)
        
        # Time step counter
        self.current_step = 0
        
        return self._get_state()
    
    def _generate_tasks(self):
        """
        Generate tasks according to Poisson arrival process
        Returns: List of tasks, where each task is (D_n, T_max_n) or None
        """
        tasks = []
        lambda_u = self.net_cfg['task_arrival_rate']
        
        for n in range(self.num_vehicles):
            # Poisson process: probability of task arrival
            if np.random.random() < lambda_u:
                # Generate task data size (in bits)
                mean_size = self.net_cfg['task_data_size_mean']
                std_size = self.net_cfg['task_data_size_std']
                data_size = np.random.normal(mean_size, std_size)
                data_size = max(0, data_size)  # Ensure non-negative
                
                # Maximum tolerable latency
                max_latency = self.net_cfg['task_max_latency']
                
                tasks.append((data_size, max_latency))
            else:
                tasks.append(None)
        
        return tasks
    
    def _compute_path_loss_a2g(self, vehicle_idx):
        """
        Compute Air-to-Ground path loss (equation 3)
        
        Args:
            vehicle_idx: Index of the vehicle
            
        Returns:
            Path loss in dB
        """
        # Distance between UAV and vehicle
        vehicle_pos = self.vehicle_positions[vehicle_idx]
        uav_pos_3d = np.array([self.uav_position[0], self.uav_position[1], self.uav_height])
        vehicle_pos_3d = np.array([vehicle_pos[0], vehicle_pos[1], 0])
        
        distance = np.linalg.norm(uav_pos_3d - vehicle_pos_3d)
        
        # Elevation angle
        theta_elevation = np.arcsin(self.uav_height / distance) if distance > 0 else 0
        
        # LoS probability
        a = self.net_cfg['a_param']
        b = self.net_cfg['b_param']
        theta_deg = np.degrees(theta_elevation)
        p_los = 1 / (1 + a * np.exp(-b * (theta_deg - a)))
        
        # Path loss
        f = self.net_cfg['carrier_frequency']
        c = self.net_cfg['speed_of_light']
        eta_los = self.net_cfg['eta_los']
        eta_nlos = self.net_cfg['eta_nlos']
        
        fspl = 20 * np.log10(distance * 4 * np.pi * f / c)
        path_loss = fspl + p_los * eta_los + (1 - p_los) * eta_nlos
        
        return path_loss
    
    def _compute_path_loss_g2g(self, vehicle_idx, rsu_idx):
        """
        Compute Ground-to-Ground path loss (equation 4)
        
        Args:
            vehicle_idx: Index of the vehicle
            rsu_idx: Index of the RSU
            
        Returns:
            Path loss in dB
        """
        vehicle_pos = self.vehicle_positions[vehicle_idx]
        rsu_pos = self.rsu_positions[rsu_idx]
        
        distance = np.linalg.norm(vehicle_pos - rsu_pos)
        
        f = self.net_cfg['carrier_frequency']
        c = self.net_cfg['speed_of_light']
        eta_rayleigh = self.net_cfg['eta_rayleigh']
        
        fspl = 20 * np.log10(distance * 4 * np.pi * f / c) if distance > 0 else 0
        path_loss = fspl + eta_rayleigh
        
        return path_loss
    
    def _compute_channel_gains(self):
        """
        Compute channel gains h_n,m for all vehicle-EN pairs
        
        Returns:
            Channel gains matrix of shape (num_vehicles, num_ens)
        """
        channel_gains = np.zeros((self.num_vehicles, self.num_ens))
        
        for n in range(self.num_vehicles):
            # Channel gains to RSUs
            for m in range(self.num_rsus):
                path_loss_db = self._compute_path_loss_g2g(n, m)
                channel_gains[n, m] = 10 ** (-path_loss_db / 10)
            
            # Channel gain to UAV
            path_loss_db = self._compute_path_loss_a2g(n)
            channel_gains[n, self.num_rsus] = 10 ** (-path_loss_db / 10)
        
        return channel_gains
    
    def _compute_transmission_rate(self, vehicle_idx, en_idx, num_associated_vehicles):
        """
        Compute transmission rate R_n,m (equation 7)
        
        Args:
            vehicle_idx: Index of the vehicle
            en_idx: Index of the edge node (0 to M-1 for RSUs, M for UAV)
            num_associated_vehicles: Number of vehicles associated with this EN
            
        Returns:
            Transmission rate in bps
        """
        # Bandwidth allocation (equal sharing)
        if en_idx < self.num_rsus:
            total_bandwidth = self.net_cfg['rsu_bandwidth']
        else:
            total_bandwidth = self.net_cfg['uav_bandwidth']
        
        beta = 1.0 / num_associated_vehicles if num_associated_vehicles > 0 else 0
        allocated_bandwidth = beta * total_bandwidth
        
        # Channel gain
        channel_gains = self._compute_channel_gains()
        h_nm = channel_gains[vehicle_idx, en_idx]
        
        # Transmit power
        p_n = self.net_cfg['vehicle_transmit_power']
        
        # Noise power
        N0 = self.net_cfg['awgn_power_density']
        noise_power = beta * N0 * total_bandwidth
        
        # Shannon capacity
        sinr = p_n * h_nm / noise_power if noise_power > 0 else 0
        rate = allocated_bandwidth * np.log2(1 + sinr)
        
        return rate
    
    def _compute_local_latency(self, vehicle_idx, offload_ratio):
        """
        Compute local computation latency (equations 10-12)
        
        Args:
            vehicle_idx: Index of the vehicle
            offload_ratio: Offloading ratio o_n
            
        Returns:
            Local service latency
        """
        task = self.current_tasks[vehicle_idx]
        if task is None:
            return 0
        
        data_size, _ = task
        local_portion = (1 - offload_ratio) * data_size
        
        # Queue delay
        F_n = self.net_cfg['vehicle_compute_capacity']
        X_c = self.net_cfg['cpu_cycles_per_bit']
        Q_n = self.vehicle_queues[vehicle_idx]
        
        tau_queue = X_c * Q_n / F_n
        
        # Computation delay
        tau_comp = (1 - offload_ratio) * X_c * data_size / F_n
        
        return tau_queue + tau_comp
    
    def _compute_offload_latency(self, vehicle_idx, en_idx, offload_ratio, num_associated):
        """
        Compute offloading latency (equations 14-17)
        
        Args:
            vehicle_idx: Index of the vehicle
            en_idx: Index of the edge node
            offload_ratio: Offloading ratio o_n
            num_associated: Number of vehicles associated with this EN
            
        Returns:
            Offloading service latency
        """
        task = self.current_tasks[vehicle_idx]
        if task is None or offload_ratio == 0:
            return 0
        
        data_size, _ = task
        offload_portion = offload_ratio * data_size
        
        # Transmission delay
        rate = self._compute_transmission_rate(vehicle_idx, en_idx, num_associated)
        tau_trans = offload_portion / rate if rate > 0 else float('inf')
        
        # Queue delay at EN
        if en_idx < self.num_rsus:
            F_m = self.net_cfg['rsu_compute_capacity']
        else:
            F_m = self.net_cfg['uav_compute_capacity']
        
        X_c = self.net_cfg['cpu_cycles_per_bit']
        Q_m = self.en_queues[en_idx]
        
        tau_queue = X_c * Q_m / F_m
        
        # Computation delay at EN
        tau_comp = offload_ratio * X_c * data_size / F_m
        
        return tau_trans + tau_queue + tau_comp
    
    def _compute_local_energy(self, vehicle_idx, offload_ratio):
        """
        Compute local computation energy (equation 13)
        
        Args:
            vehicle_idx: Index of the vehicle
            offload_ratio: Offloading ratio o_n
            
        Returns:
            Local computation energy consumption in Joules
        """
        task = self.current_tasks[vehicle_idx]
        if task is None:
            return 0
        
        data_size, _ = task
        F_n = self.net_cfg['vehicle_compute_capacity']
        X_c = self.net_cfg['cpu_cycles_per_bit']
        kappa = self.net_cfg['kappa_vehicle']
        
        energy = kappa * (F_n ** 2) * (1 - offload_ratio) * X_c * data_size
        
        return energy
    
    def _compute_offload_energy(self, vehicle_idx, en_idx, offload_ratio, num_associated):
        """
        Compute offloading energy (equation 18)
        
        Args:
            vehicle_idx: Index of the vehicle
            en_idx: Index of the edge node
            offload_ratio: Offloading ratio o_n
            num_associated: Number of vehicles associated with this EN
            
        Returns:
            Offloading energy consumption in Joules
        """
        task = self.current_tasks[vehicle_idx]
        if task is None or offload_ratio == 0:
            return 0
        
        data_size, _ = task
        offload_portion = offload_ratio * data_size
        
        # Transmission time
        rate = self._compute_transmission_rate(vehicle_idx, en_idx, num_associated)
        tau_trans = offload_portion / rate if rate > 0 else 0
        
        # Transmission energy
        p_n = self.net_cfg['vehicle_transmit_power']
        energy = p_n * tau_trans
        
        return energy
    
    def _compute_uav_energy(self, velocity):
        """
        Compute UAV propulsion energy consumption (equation 20)
        
        Args:
            velocity: UAV flight velocity
            
        Returns:
            UAV energy consumption in Joules
        """
        P0 = self.net_cfg['P0']
        Pi = self.net_cfg['Pi']
        Utip = self.net_cfg['Utip']
        v0 = self.net_cfg['v0']
        d0 = self.net_cfg['d0']
        rho0 = self.net_cfg['rho0']
        s0 = self.net_cfg['s0']
        A0 = self.net_cfg['A0']
        tau = self.tau
        
        # Equation 20
        term1 = P0 * (1 + 3 * velocity**2 / Utip**2)
        term2 = Pi * ((1 + velocity**4 / (4 * v0**4) - velocity**2 / (2 * v0**2)) ** 0.5)
        term3 = 0.5 * d0 * rho0 * s0 * A0 * velocity**3
        
        energy = tau * (term1 + term2 + term3)
        
        return energy
    
    def _compute_rental_price(self, vehicle_idx, en_idx, offload_ratio):
        """
        Compute rental price (equation 26)
        
        Args:
            vehicle_idx: Index of the vehicle
            en_idx: Index of the edge node
            offload_ratio: Offloading ratio o_n
            
        Returns:
            Rental price
        """
        task = self.current_tasks[vehicle_idx]
        if task is None or offload_ratio == 0:
            return 0
        
        data_size, _ = task
        X_c = self.net_cfg['cpu_cycles_per_bit']
        
        if en_idx < self.num_rsus:
            price_per_cycle = self.net_cfg['rsu_rental_price']
        else:
            price_per_cycle = self.net_cfg['uav_rental_price']
        
        rental_price = price_per_cycle * offload_ratio * X_c * data_size
        
        return rental_price
    
    def _get_state(self):
        """
        Construct state vector according to equation (29)
        
        Returns:
            State vector as numpy array
        """
        state = []
        
        # W(t): Vehicle and UAV positions (2(N+1) dimensions)
        for pos in self.vehicle_positions:
            state.extend(pos)
        state.extend(self.uav_position)
        
        # U(t): Task information (2N dimensions)
        for task in self.current_tasks:
            if task is not None:
                state.extend([task[0], task[1]])  # (D_n, T_max_n)
            else:
                state.extend([0, 0])
        
        # Y(t): Task intervals (N dimensions)
        state.extend(self.task_intervals)
        
        # Q(t): Queue backlogs (N + M + 1 dimensions)
        state.extend(self.vehicle_queues)
        state.extend(self.en_queues)
        
        # H(t): Channel gains (N(M+1) dimensions)
        channel_gains = self._compute_channel_gains()
        state.extend(channel_gains.flatten())
        
        # E_res_u: UAV residual energy (1 dimension)
        state.append(self.uav_residual_energy)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Execute one time step
        
        Args:
            action: Action vector [v_u, theta_u, alpha_1, ..., alpha_N]
            
        Returns:
            next_state, reward, done, info
        """
        # Parse action
        v_u = action[0]
        theta_u = action[1]
        alphas = action[2:]
        
        # Update UAV position (equations 1-2)
        self.uav_position[0] += self.tau * v_u * np.cos(theta_u)
        self.uav_position[1] += self.tau * v_u * np.sin(theta_u)
        
        # Keep UAV within map bounds
        area_width, area_height = self.map_cfg['area']
        self.uav_position[0] = np.clip(self.uav_position[0], 0, area_width)
        self.uav_position[1] = np.clip(self.uav_position[1], 0, area_height)
        
        # Update vehicle positions
        for n in range(self.num_vehicles):
            self.vehicle_positions[n, 0] += self.tau * self.vehicle_velocities[n] * np.cos(self.vehicle_angles[n])
            self.vehicle_positions[n, 1] += self.tau * self.vehicle_velocities[n] * np.sin(self.vehicle_angles[n])
            
            # Wrap around or reflect at boundaries
            self.vehicle_positions[n, 0] = np.clip(self.vehicle_positions[n, 0], 0, area_width)
            self.vehicle_positions[n, 1] = np.clip(self.vehicle_positions[n, 1], 0, area_height)
        
        # Decode user association from alpha (equation 30-31)
        associations = np.zeros((self.num_vehicles, self.num_ens), dtype=int)
        offload_ratios = np.zeros(self.num_vehicles)
        
        for n in range(self.num_vehicles):
            alpha_n = alphas[n]
            
            if alpha_n <= 1/3:
                # Local computation
                associations[n, :] = 0
                offload_ratios[n] = 0
            elif alpha_n <= 2/3:
                # Offload to RSU
                # Find closest RSU
                vehicle_pos = self.vehicle_positions[n]
                distances = [np.linalg.norm(vehicle_pos - rsu_pos) 
                           for rsu_pos in self.rsu_positions]
                closest_rsu = np.argmin(distances)
                associations[n, closest_rsu] = 1
                offload_ratios[n] = 3 * alpha_n - 1
            else:
                # Offload to UAV
                associations[n, self.num_rsus] = 1
                offload_ratios[n] = 1  # Full offload to UAV
        
        # Count associations per EN
        en_associations = associations.sum(axis=0)
        
        # Compute costs for each vehicle
        total_cost = 0
        penalty = 0
        
        for n in range(self.num_vehicles):
            task = self.current_tasks[n]
            if task is None:
                continue
            
            # Find associated EN
            en_idx = np.argmax(associations[n]) if associations[n].sum() > 0 else -1
            offload_ratio = offload_ratios[n]
            
            # Compute latencies
            tau_local = self._compute_local_latency(n, offload_ratio)
            
            if en_idx >= 0 and offload_ratio > 0:
                num_assoc = en_associations[en_idx]
                tau_offload = self._compute_offload_latency(n, en_idx, offload_ratio, num_assoc)
            else:
                tau_offload = 0
            
            # Total execution time
            tau_total = max(tau_local, tau_offload)
            
            # Update AoI (equation 24)
            aoi_cost = tau_total + self.task_intervals[n]
            
            # Compute energy consumption (equation 25)
            energy_local = self._compute_local_energy(n, offload_ratio)
            if en_idx >= 0 and offload_ratio > 0:
                num_assoc = en_associations[en_idx]
                energy_offload = self._compute_offload_energy(n, en_idx, offload_ratio, num_assoc)
            else:
                energy_offload = 0
            energy_total = energy_local + energy_offload
            
            # Compute rental price (equation 26)
            if en_idx >= 0 and offload_ratio > 0:
                rental_price = self._compute_rental_price(n, en_idx, offload_ratio)
            else:
                rental_price = 0
            
            # Weighted cost (equation 27)
            gamma_A = self.net_cfg['gamma_A']
            gamma_E = self.net_cfg['gamma_E']
            gamma_P = self.net_cfg['gamma_P']
            
            vehicle_cost = gamma_A * aoi_cost + gamma_E * energy_total + gamma_P * rental_price
            total_cost += vehicle_cost
            
            # Check constraints and add penalties (equations 34-35)
            # Latency constraint violation (28i)
            data_size, max_latency = task
            if tau_total > max_latency:
                penalty += self.net_cfg['c2']
            
            # Update queues (equations 8-9)
            F_n = self.net_cfg['vehicle_compute_capacity']
            X_c = self.net_cfg['cpu_cycles_per_bit']
            D_comp_n = min(self.tau * F_n / X_c, self.vehicle_queues[n])
            self.vehicle_queues[n] = max(self.vehicle_queues[n] - D_comp_n, 0) + (1 - offload_ratio) * data_size
            
            if en_idx >= 0 and offload_ratio > 0:
                if en_idx < self.num_rsus:
                    F_m = self.net_cfg['rsu_compute_capacity']
                else:
                    F_m = self.net_cfg['uav_compute_capacity']
                
                D_comp_m = min(self.tau * F_m / X_c, self.en_queues[en_idx])
                self.en_queues[en_idx] = max(self.en_queues[en_idx] - D_comp_m, 0) + offload_ratio * data_size
        
        # Check association constraints (28f, 28g) and add penalty (equation 34)
        for m in range(self.num_rsus):
            if en_associations[m] > self.net_cfg['rsu_max_associations']:
                penalty += self.net_cfg['c1']
        
        if en_associations[self.num_rsus] > self.net_cfg['uav_max_associations']:
            penalty += self.net_cfg['c1']
        
        # Update UAV energy
        uav_energy_consumed = self._compute_uav_energy(v_u)
        self.uav_residual_energy -= uav_energy_consumed
        
        # Energy constraint penalty (equation 36)
        if self.uav_residual_energy < 0:
            penalty += self.net_cfg['c3']

        # Compute reward (equation 33)
        reward = -(total_cost + penalty)
        
        # Update task intervals and generate new tasks
        self.task_intervals += self.tau
        self.current_tasks = self._generate_tasks()
        
        # Update time step
        self.current_step += 1
        done = self.current_step >= self.td3_cfg['max_steps']
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'total_cost': total_cost,
            'penalty': penalty,
            'uav_energy': self.uav_residual_energy,
        }
        
        return next_state, reward, done, info
