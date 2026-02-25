"""
UAV-Assisted VEC Environment with MIMO-NOMA Channel
Integrates the NOMA channel model from:
  "Decentralized Power Allocation for MIMO-NOMA Vehicular Edge Computing
   Based on Deep Reinforcement Learning" (Zhu et al., 2021)

Key additions over base environment:
  - MIMO-NOMA channel with ZF detection at BS/UAV
  - Small-scale Rayleigh fading via AR model (Jake's fading spectrum)
  - Large-scale path loss based on vehicle mobility
  - Per-vehicle SINR computed via ZF detector
  - Buffer-based task processing model (eqs. 15-18 from paper)
  - NOMA uplink transmission rate replacing simple Shannon capacity
"""

import numpy as np
import gym
from gym import spaces
from scipy.special import j0  # Zeroth-order Bessel function


class UAVAssistedVECEnvNOMA(gym.Env):
    """
    UAV-Assisted VEC Environment with MIMO-NOMA channel model.

    NOMA Channel additions (from Zhu et al. 2021):
      - BS/UAV equipped with N antennas (MIMO)
      - All vehicles share the same spectrum (NOMA)
      - ZF detector separates per-vehicle signals
      - Jake's Doppler model for small-scale fading AR process
      - SINR replaces simple SNR in transmission rate calculation
    """

    def __init__(self, config):
        super(UAVAssistedVECEnvNOMA, self).__init__()

        self.config = config
        self.td3_cfg = config['td3']
        self.net_cfg = config['network']
        self.map_cfg = config['map']
        self.noma_cfg = config.get('noma', {})
        self.num_vehicles = config['num_vehicles']

        # Base parameters
        self.tau = self.net_cfg['time_slot_duration']
        self.num_rsus = self.map_cfg['num_rsus']
        self.num_ens = self.num_rsus + 1  # RSUs + UAV

        # RSU positions
        self.rsu_positions = np.array(self.map_cfg['rsu_positions'])

        # ---------------------------------------------------------------
        # NOMA / MIMO parameters (Section III of Zhu et al. 2021)
        # ---------------------------------------------------------------
        # Number of antennas at BS (N in paper, eq. 4)
        self.num_bs_antennas = self.noma_cfg.get('num_bs_antennas', 4)

        # Number of antennas at UAV (also MIMO-capable)
        self.num_uav_antennas = self.noma_cfg.get('num_uav_antennas', 4)

        # Reference channel power gain at 1 m (h_r, eq. 6)
        self.h_r = self.noma_cfg.get('h_r', 10 ** (-3))  # -30 dB

        # Path-loss exponent (eta, eq. 6)
        self.path_loss_exponent = self.noma_cfg.get('path_loss_exponent', 2.0)

        # Carrier wavelength (Lambda, eq. 8)
        self.wavelength = self.noma_cfg.get('wavelength', 0.075)  # ~4 GHz

        # AWGN noise variance at receiver (sigma^2_R, eq. 4 & 14)
        self.noise_variance = self.noma_cfg.get('noise_variance', 1e-9)

        # Maximum offloading power per vehicle (P_max,o)
        self.p_max_offload = self.net_cfg.get('vehicle_transmit_power', 1.0)

        # AR model decay parameter for small-scale fading (rho_m, eq. 7)
        # Computed per vehicle based on Doppler; stored per-vehicle each step
        self._rho = np.ones(self.num_vehicles)  # will be recomputed each step

        # ---------------------------------------------------------------
        # NOMA state memory: small-scale fading and SINR from last slot
        # ---------------------------------------------------------------
        # h^s_m(t): small-scale fading vectors, shape (N_vehicles, N_antennas)
        # We track both RSU and UAV fading channels separately
        self._h_small_rsu = None   # (num_vehicles, num_bs_antennas)  -- shared across RSUs for simplicity
        self._h_small_uav = None   # (num_vehicles, num_uav_antennas)

        # gamma_m(t-1): SINR from previous slot, used in state (eq. 19)
        self._sinr_prev_rsu = np.zeros(self.num_vehicles)
        self._sinr_prev_uav = np.zeros(self.num_vehicles)

        # Buffer lengths B_m(t) — replaces queue model from base env (eq. 15)
        self.vehicle_buffers = np.zeros(self.num_vehicles)

        # ---------------------------------------------------------------
        # Spaces
        # ---------------------------------------------------------------
        self._setup_spaces()
        self.reset()

    # ==================================================================
    #  Space setup
    # ==================================================================
    def _setup_spaces(self):
        """Define action and observation spaces."""
        # Action: [v_u, theta_u, alpha_1, ..., alpha_N]
        action_low = np.array([0, 0] + [0] * self.num_vehicles, dtype=np.float32)
        action_high = np.array(
            [self.net_cfg['uav_max_velocity'], 2 * np.pi] + [1] * self.num_vehicles,
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # State: positions + tasks + intervals + queues/buffers + NOMA channels + UAV energy
        # Extended with NOMA-specific terms:
        #   SINR_prev per vehicle per EN  :  num_vehicles * num_ens
        #   Buffer lengths                :  num_vehicles   (replaces part of Q(t))
        state_dim = (
            2 * (self.num_vehicles + 1) +       # W(t): vehicle + UAV positions
            2 * self.num_vehicles +              # U(t): task info (D_n, T_max_n)
            self.num_vehicles +                  # Y(t): task intervals
            self.num_vehicles + self.num_ens +   # Q(t): vehicle queues + EN queues
            self.num_vehicles * self.num_ens +   # H(t): channel gains (large-scale)
            self.num_vehicles * self.num_ens +   # SINR_prev: NOMA SINR from last slot
            self.num_vehicles +                  # B(t): buffer lengths (NOMA buffer model)
            1                                    # E_res_u: residual UAV energy
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

    # ==================================================================
    #  Reset
    # ==================================================================
    def reset(self):
        """Reset environment to initial state."""
        area_width, area_height = self.map_cfg['area']

        # Vehicle positions and kinematics
        self.vehicle_positions = np.random.uniform(
            low=[0, 0], high=[area_width, area_height],
            size=(self.num_vehicles, 2)
        )
        v_min, v_max = self.net_cfg['vehicle_velocity_range']
        self.vehicle_velocities = np.random.uniform(v_min, v_max, self.num_vehicles)
        self.vehicle_angles = np.random.uniform(0, 2 * np.pi, self.num_vehicles)

        # UAV state
        self.uav_position = np.array([area_width / 2, area_height / 2], dtype=np.float64)
        self.uav_height = self.net_cfg['uav_height']
        self.uav_residual_energy = self.net_cfg['uav_max_energy']

        # Legacy queue backlogs (EN queues still used for RSU)
        self.vehicle_queues = np.zeros(self.num_vehicles)
        self.en_queues = np.zeros(self.num_ens)

        # NOMA buffer model (eq. 15): B_m(t)
        self.vehicle_buffers = np.zeros(self.num_vehicles)

        # Task state
        self.task_intervals = np.zeros(self.num_vehicles)
        self.current_tasks = self._generate_tasks()

        # AoI
        self.aoi = np.zeros(self.num_vehicles)

        # ---------------------------------------------------------------
        # Initialise NOMA small-scale fading channels (complex Gaussian)
        # h^s_m(0) ~ CN(0, I_N)  (Zhu eq. paper section V-A)
        # ---------------------------------------------------------------
        self._h_small_rsu = (
            np.random.randn(self.num_vehicles, self.num_bs_antennas) +
            1j * np.random.randn(self.num_vehicles, self.num_bs_antennas)
        ) / np.sqrt(2)

        self._h_small_uav = (
            np.random.randn(self.num_vehicles, self.num_uav_antennas) +
            1j * np.random.randn(self.num_vehicles, self.num_uav_antennas)
        ) / np.sqrt(2)

        # Initial SINR (computed from initial channel state)
        self._sinr_prev_rsu = np.zeros(self.num_vehicles)
        self._sinr_prev_uav = np.zeros(self.num_vehicles)

        # Compute initial SINR for first state observation
        sinr_rsu, sinr_uav = self._compute_noma_sinr()
        self._sinr_prev_rsu = sinr_rsu
        self._sinr_prev_uav = sinr_uav

        self.current_step = 0
        return self._get_state()

    # ==================================================================
    #  Task generation
    # ==================================================================
    def _generate_tasks(self):
        """Generate tasks via Poisson arrival process."""
        tasks = []
        lambda_u = self.net_cfg['task_arrival_rate']
        for n in range(self.num_vehicles):
            if np.random.random() < lambda_u:
                mean_size = self.net_cfg['task_data_size_mean']
                std_size = self.net_cfg['task_data_size_std']
                data_size = max(0, np.random.normal(mean_size, std_size))
                max_latency = self.net_cfg['task_max_latency']
                tasks.append((data_size, max_latency))
            else:
                tasks.append(None)
        return tasks

    # ==================================================================
    #  NOMA Channel Model  (Zhu et al. 2021, Section III-B)
    # ==================================================================

    def _closest_rsu(self, vehicle_idx):
        """Return the index of the RSU closest to vehicle n."""
        distances = [
            np.linalg.norm(self.vehicle_positions[vehicle_idx] - rsu_pos)
            for rsu_pos in self.rsu_positions
        ]
        return int(np.argmin(distances))

    def _compute_large_scale_path_loss(self, vehicle_idx, en_idx):
        """
        Large-scale fading coefficient h^p_m(t) (eq. 6).

        For RSUs: uses ground-to-ground distance.
        For UAV (en_idx == num_rsus): uses 3D distance with UAV height.

        Returns: h^p_m  (scalar, linear scale)
        """
        vehicle_pos = self.vehicle_positions[vehicle_idx]

        if en_idx < self.num_rsus:
            rsu_pos = self.rsu_positions[en_idx]
            distance = np.linalg.norm(vehicle_pos - rsu_pos)
            distance = max(distance, 1.0)  # avoid log(0)
        else:
            # UAV as edge node
            uav_pos_3d = np.array([self.uav_position[0], self.uav_position[1], self.uav_height])
            vehicle_pos_3d = np.array([vehicle_pos[0], vehicle_pos[1], 0.0])
            distance = np.linalg.norm(uav_pos_3d - vehicle_pos_3d)
            distance = max(distance, 1.0)

        # eq. 6:  h^p_m = h_r / d^eta
        h_p = self.h_r / (distance ** self.path_loss_exponent)
        return h_p

    def _compute_doppler_rho(self, vehicle_idx, en_idx):
        """
        Normalized channel correlation coefficient rho_m between consecutive
        slots (eq. 7-9 of Zhu et al. 2021).

        rho_m = J_0(2*pi*f^m_d * tau_0)

        where f^m_d = (v_m / Lambda) * cos(Theta)
        and   cos(Theta) = x0 · (P_BS - P_m) / ||P_BS - P_m||
        """
        vehicle_pos = self.vehicle_positions[vehicle_idx]
        v_m = self.vehicle_velocities[vehicle_idx]

        if en_idx < self.num_rsus:
            # RSU as receiver — use closest RSU position as "BS"
            rx_pos = np.array([self.rsu_positions[en_idx][0],
                                self.rsu_positions[en_idx][1],
                                0.0])
        else:
            rx_pos = np.array([self.uav_position[0],
                                self.uav_position[1],
                                self.uav_height])

        p_m = np.array([vehicle_pos[0], vehicle_pos[1], 0.0])
        direction_vec = rx_pos - p_m
        dist = np.linalg.norm(direction_vec)

        if dist < 1e-6:
            cos_theta = 0.0
        else:
            # Moving direction unit vector (eq. 9, x0 = (1,0,0) = east)
            move_dir = np.array([np.cos(self.vehicle_angles[vehicle_idx]),
                                  np.sin(self.vehicle_angles[vehicle_idx]),
                                  0.0])
            cos_theta = np.dot(move_dir, direction_vec / dist)

        # Doppler frequency (eq. 8)
        f_d = (v_m / self.wavelength) * cos_theta

        # Jake's model (eq. 7): rho = J_0(2*pi*f_d*tau_0)
        rho = float(j0(2 * np.pi * f_d * self.tau))
        rho = np.clip(rho, -1.0, 1.0)
        return rho

    def _update_small_scale_fading(self):
        """
        Update small-scale Rayleigh fading via AR model (eq. 7):
            h^s_m(t) = rho_m * h^s_m(t-1) + sqrt(1 - rho_m^2) * e(t)

        where e(t) ~ CN(0, I_N).
        Applied independently to RSU and UAV channels.
        """
        for n in range(self.num_vehicles):
            # RSU channel update — use the RSU closest to vehicle n for Doppler calculation
            closest_rsu_idx = self._closest_rsu(n)
            rho_rsu = self._compute_doppler_rho(n, en_idx=closest_rsu_idx)
            e_rsu = (
                np.random.randn(self.num_bs_antennas) +
                1j * np.random.randn(self.num_bs_antennas)
            ) / np.sqrt(2)
            self._h_small_rsu[n] = (
                rho_rsu * self._h_small_rsu[n] +
                np.sqrt(max(1 - rho_rsu ** 2, 0)) * e_rsu
            )

            # UAV channel update
            rho_uav = self._compute_doppler_rho(n, en_idx=self.num_rsus)
            e_uav = (
                np.random.randn(self.num_uav_antennas) +
                1j * np.random.randn(self.num_uav_antennas)
            ) / np.sqrt(2)
            self._h_small_uav[n] = (
                rho_uav * self._h_small_uav[n] +
                np.sqrt(max(1 - rho_uav ** 2, 0)) * e_uav
            )

    def _build_channel_matrix(self, en_idx, offload_powers):
        """
        Build the MIMO channel matrix H(t) ∈ C^{N x M} for a given EN (eq. 4-5).
        H[:, m] = h^s_m * sqrt(h^p_m)

        Args:
            en_idx: Edge node index (RSU or UAV)
            offload_powers: array of shape (num_vehicles,) — per-vehicle offload power

        Returns:
            H: complex array (num_antennas, num_vehicles)
            h_p: large-scale gains array (num_vehicles,)
        """
        if en_idx < self.num_rsus:
            num_antennas = self.num_bs_antennas
            h_small = self._h_small_rsu  # (N_veh, N_ant)
        else:
            num_antennas = self.num_uav_antennas
            h_small = self._h_small_uav

        H = np.zeros((num_antennas, self.num_vehicles), dtype=complex)
        h_p_arr = np.zeros(self.num_vehicles)

        for n in range(self.num_vehicles):
            h_p = self._compute_large_scale_path_loss(n, en_idx)
            h_p_arr[n] = h_p
            # eq. 5: h_m = h^s_m * sqrt(h^p_m)
            H[:, n] = h_small[n] * np.sqrt(h_p)

        return H, h_p_arr

    def _zf_sinr(self, H, offload_powers):
        """
        Compute per-vehicle SINR at receiver using Zero-Forcing detector (eq. 10-14).

        ZF pseudo-inverse: H_dag = (H^H H)^{-1} H^H
        SINR_m = p_m,o / (||g^H_m||^2 * sigma^2_R)   (eq. 14)

        Args:
            H: channel matrix (num_antennas, num_vehicles), complex
            offload_powers: (num_vehicles,) offloading power for each VU

        Returns:
            sinr: array (num_vehicles,) with per-vehicle SINR
        """
        N, M = H.shape

        if M == 0 or N < M:
            # Under-determined: cannot do ZF — fall back to single-vehicle case
            sinr = np.zeros(M)
            for m in range(M):
                h_m = H[:, m]
                h_norm_sq = np.real(np.dot(h_m.conj(), h_m))
                sinr[m] = (offload_powers[m] * h_norm_sq / self.noise_variance
                           if self.noise_variance > 0 else 0)
            return sinr

        # ZF detector: H_dag = (H^H H)^{-1} H^H  (eq. 10)
        HH = H.conj().T @ H  # (M, M)
        try:
            HH_inv = np.linalg.inv(HH + 1e-12 * np.eye(M))  # regularized for stability
        except np.linalg.LinAlgError:
            HH_inv = np.linalg.pinv(HH)

        H_dag = HH_inv @ H.conj().T  # (M, N)  — rows are g^H_m

        sinr = np.zeros(M)
        for m in range(M):
            g_m = H_dag[m, :]           # row m of ZF detector (eq. 10)
            g_norm_sq = np.real(np.dot(g_m.conj(), g_m))  # ||g^H_m||^2
            # eq. 14: gamma_m = p_m,o / (||g^H_m||^2 * sigma^2_R)
            if g_norm_sq > 0 and self.noise_variance > 0:
                sinr[m] = offload_powers[m] / (g_norm_sq * self.noise_variance)
            else:
                sinr[m] = 0.0

        return sinr

    def _compute_noma_sinr(self, offload_powers=None):
        """
        Compute NOMA SINR for all vehicles at RSU cluster and UAV.

        We treat all RSUs as sharing a combined antenna array for simplicity
        (one SINR value per vehicle per EN type).

        Args:
            offload_powers: (num_vehicles,) per-vehicle offload power.
                            Defaults to max offload power for all vehicles.

        Returns:
            sinr_rsu: (num_vehicles,) SINR at RSU
            sinr_uav: (num_vehicles,) SINR at UAV
        """
        if offload_powers is None:
            offload_powers = np.full(self.num_vehicles, self.p_max_offload)

        # RSU SINR — build channel matrix using each vehicle's closest RSU.
        # We still produce a single (num_antennas, num_vehicles) matrix H_rsu,
        # but column n uses the path loss to vehicle n's closest RSU rather
        # than always RSU 0.  The ZF detector then separates all vehicles
        # simultaneously, which is correct for a NOMA uplink.
        H_rsu = np.zeros((self.num_bs_antennas, self.num_vehicles), dtype=complex)
        for n in range(self.num_vehicles):
            closest_rsu_idx = self._closest_rsu(n)
            h_p = self._compute_large_scale_path_loss(n, closest_rsu_idx)
            H_rsu[:, n] = self._h_small_rsu[n] * np.sqrt(h_p)
        sinr_rsu = self._zf_sinr(H_rsu, offload_powers)

        # UAV SINR (unchanged — there is only one UAV)
        H_uav, _ = self._build_channel_matrix(self.num_rsus, offload_powers)
        sinr_uav = self._zf_sinr(H_uav, offload_powers)

        return sinr_rsu, sinr_uav

    def _compute_noma_transmission_rate(self, vehicle_idx, en_idx, offload_power=None):
        """
        Compute NOMA uplink transmission rate for vehicle n to EN m (eq. 18).

        Rate = tau_0 * W * log2(1 + SINR_m(t-1))  [bits per slot]
        (We use SINR from PREVIOUS slot as it is only available to VU at current slot.)

        Args:
            vehicle_idx: Vehicle index n
            en_idx: Edge node index m
            offload_power: offload power for this vehicle (used for SINR computation)

        Returns:
            rate (bits/slot), sinr (scalar)
        """
        if en_idx < self.num_rsus:
            # SINR stored in _sinr_prev_rsu was computed using each vehicle's
            # closest RSU, so it already reflects the correct RSU for vehicle n.
            sinr = self._sinr_prev_rsu[vehicle_idx]
            bandwidth = self.net_cfg.get('rsu_bandwidth', 1e6)
        else:
            sinr = self._sinr_prev_uav[vehicle_idx]
            bandwidth = self.net_cfg.get('uav_bandwidth', 1e6)

        # Shannon capacity (eq. 18): d_m,o = tau_0 * W * log2(1 + gamma_m)
        rate = self.tau * bandwidth * np.log2(1 + max(sinr, 0))
        return rate, sinr

    # ==================================================================
    #  Buffer-Based Computation Model  (Zhu et al. 2021, Section III-C)
    # ==================================================================

    def _update_vehicle_buffer(self, vehicle_idx, task_offloaded_bits, task_local_bits, task_arrived_bits):
        """
        Update buffer for vehicle n (eq. 15):
            B_m(t) = max(0, B_m(t-1) - (d_m,o + d_m,l)) + a_m(t-1)

        Args:
            vehicle_idx: Vehicle index
            task_offloaded_bits: d_m,o(t-1) - bits processed via offloading
            task_local_bits:     d_m,l(t-1) - bits processed locally
            task_arrived_bits:   a_m(t-1)   - bits that arrived in previous slot
        """
        B_prev = self.vehicle_buffers[vehicle_idx]
        departed = task_offloaded_bits + task_local_bits
        self.vehicle_buffers[vehicle_idx] = max(0, B_prev - departed) + task_arrived_bits

    def _compute_local_processed_bits(self, vehicle_idx, local_power):
        """
        Local execution: bits processed in one slot (eq. 16-17).
            f_m = (p_m,l / kappa)^{1/3}
            d_m,l = tau_0 * f_m / L_m

        Args:
            vehicle_idx: Vehicle index
            local_power: Local execution power p_m,l

        Returns:
            bits processed locally in this slot
        """
        kappa = self.net_cfg.get('kappa_vehicle', 1e-28)
        L_m = self.net_cfg.get('cpu_cycles_per_bit', 500)
        F_max = self.net_cfg.get('vehicle_compute_capacity', 2.15e9)

        # eq. 17: f_m = (p_m,l / kappa)^{1/3}
        f_m = (local_power / kappa) ** (1 / 3) if local_power > 0 else 0
        f_m = min(f_m, F_max)

        # eq. 16: d_m,l = tau_0 * f_m / L_m
        d_local = self.tau * f_m / L_m
        return d_local, f_m

    # ==================================================================
    #  Legacy path-loss helpers (kept for RSU G2G, UAV A2G)
    # ==================================================================

    def _compute_path_loss_a2g(self, vehicle_idx):
        """Air-to-Ground path loss (eq. 3, base env)."""
        vehicle_pos = self.vehicle_positions[vehicle_idx]
        uav_pos_3d = np.array([self.uav_position[0], self.uav_position[1], self.uav_height])
        vehicle_pos_3d = np.array([vehicle_pos[0], vehicle_pos[1], 0.0])
        distance = np.linalg.norm(uav_pos_3d - vehicle_pos_3d)
        if distance < 1.0:
            distance = 1.0
        theta_elevation = np.arcsin(self.uav_height / distance)
        a = self.net_cfg.get('a_param', 9.61)
        b = self.net_cfg.get('b_param', 0.16)
        theta_deg = np.degrees(theta_elevation)
        p_los = 1 / (1 + a * np.exp(-b * (theta_deg - a)))
        f = self.net_cfg.get('carrier_frequency', 2e9)
        c = self.net_cfg.get('speed_of_light', 3e8)
        eta_los = self.net_cfg.get('eta_los', 1.0)
        eta_nlos = self.net_cfg.get('eta_nlos', 20.0)
        fspl = 20 * np.log10(distance * 4 * np.pi * f / c)
        path_loss = fspl + p_los * eta_los + (1 - p_los) * eta_nlos
        return path_loss

    def _compute_path_loss_g2g(self, vehicle_idx, rsu_idx):
        """Ground-to-Ground path loss (eq. 4, base env)."""
        vehicle_pos = self.vehicle_positions[vehicle_idx]
        rsu_pos = self.rsu_positions[rsu_idx]
        distance = np.linalg.norm(vehicle_pos - rsu_pos)
        if distance < 1.0:
            distance = 1.0
        f = self.net_cfg.get('carrier_frequency', 2e9)
        c = self.net_cfg.get('speed_of_light', 3e8)
        eta_rayleigh = self.net_cfg.get('eta_rayleigh', 0.0)
        fspl = 20 * np.log10(distance * 4 * np.pi * f / c)
        return fspl + eta_rayleigh

    def _compute_channel_gains(self):
        """
        Large-scale channel gains matrix (num_vehicles, num_ens).
        Used in the state vector H(t).
        """
        channel_gains = np.zeros((self.num_vehicles, self.num_ens))
        for n in range(self.num_vehicles):
            for m in range(self.num_rsus):
                pl_db = self._compute_path_loss_g2g(n, m)
                channel_gains[n, m] = 10 ** (-pl_db / 10)
            pl_db = self._compute_path_loss_a2g(n)
            channel_gains[n, self.num_rsus] = 10 ** (-pl_db / 10)
        return channel_gains

    # ==================================================================
    #  Energy computations
    # ==================================================================

    def _compute_local_energy(self, vehicle_idx, local_power, f_m):
        """
        Local computation energy (eq. 13, base env):
            E_local = kappa * f_m^2 * d_m,l * L_m
        """
        task = self.current_tasks[vehicle_idx]
        if task is None:
            return 0
        kappa = self.net_cfg.get('kappa_vehicle', 1e-28)
        L_m = self.net_cfg.get('cpu_cycles_per_bit', 500)
        d_local, _ = self._compute_local_processed_bits(vehicle_idx, local_power)
        energy = kappa * (f_m ** 2) * d_local * L_m
        return energy

    def _compute_offload_energy(self, vehicle_idx, offload_power, tau_trans):
        """
        Offloading transmission energy:
            E_offload = p_m,o * tau_trans
        """
        return offload_power * tau_trans

    def _compute_uav_energy(self, velocity):
        """UAV propulsion energy (eq. 20, base env)."""
        P0 = self.net_cfg.get('P0', 79.86)
        Pi = self.net_cfg.get('Pi', 88.63)
        Utip = self.net_cfg.get('Utip', 120.0)
        v0 = self.net_cfg.get('v0', 4.03)
        d0 = self.net_cfg.get('d0', 0.6)
        rho0 = self.net_cfg.get('rho0', 1.225)
        s0 = self.net_cfg.get('s0', 0.05)
        A0 = self.net_cfg.get('A0', 0.503)
        term1 = P0 * (1 + 3 * velocity ** 2 / Utip ** 2)
        term2 = Pi * ((1 + velocity ** 4 / (4 * v0 ** 4) - velocity ** 2 / (2 * v0 ** 2)) ** 0.5)
        term3 = 0.5 * d0 * rho0 * s0 * A0 * velocity ** 3
        return self.tau * (term1 + term2 + term3)

    def _compute_rental_price(self, vehicle_idx, en_idx, offload_bits):
        """Rental price for EN computation resources (eq. 26, base env)."""
        L_m = self.net_cfg.get('cpu_cycles_per_bit', 500)
        if en_idx < self.num_rsus:
            price_per_cycle = self.net_cfg.get('rsu_rental_price', 1e-9)
        else:
            price_per_cycle = self.net_cfg.get('uav_rental_price', 1.5e-9)
        return price_per_cycle * offload_bits * L_m

    # ==================================================================
    #  State vector
    # ==================================================================

    def _get_state(self):
        """
        Construct extended state vector with NOMA additions (based on eq. 29 + eq. 19).

        Extended state includes:
          - Original terms: W(t), U(t), Y(t), Q(t), H(t), E_res_u
          - NOMA additions: SINR_prev per vehicle-EN pair, buffer lengths B_m(t)
        """
        state = []

        # W(t): positions (2*(N+1))
        for pos in self.vehicle_positions:
            state.extend(pos)
        state.extend(self.uav_position)

        # U(t): task info (2N)
        for task in self.current_tasks:
            state.extend([task[0], task[1]] if task is not None else [0, 0])

        # Y(t): task intervals (N)
        state.extend(self.task_intervals)

        # Q(t): vehicle queues + EN queues (N + M+1)
        state.extend(self.vehicle_queues)
        state.extend(self.en_queues)

        # H(t): large-scale channel gains (N*(M+1))
        channel_gains = self._compute_channel_gains()
        state.extend(channel_gains.flatten())

        # NOMA addition: SINR_prev for RSU and UAV (N * num_ens)
        for n in range(self.num_vehicles):
            for m in range(self.num_rsus):
                state.append(self._sinr_prev_rsu[n])   # same RSU SINR for all RSUs (simplification)
            state.append(self._sinr_prev_uav[n])        # UAV SINR

        # NOMA addition: buffer lengths B_m(t) (N)
        state.extend(self.vehicle_buffers)

        # E_res_u (1)
        state.append(self.uav_residual_energy)

        return np.array(state, dtype=np.float32)

    # ==================================================================
    #  Step
    # ==================================================================

    def step(self, action):
        """
        Execute one time step with NOMA channel model.

        Action: [v_u, theta_u, alpha_1, ..., alpha_N]
          alpha_n encodes offload destination and ratio (same as base env).

        NOMA integration:
          1. Update small-scale fading channels (AR model, eq. 7)
          2. Compute ZF-based SINR for all vehicle-EN pairs (eq. 14)
          3. Use SINR from PREVIOUS slot for rate calculation (eq. 18)
          4. Update buffers (eq. 15) instead of/alongside queue model
        """
        # ---- Parse action ----
        v_u = float(action[0])
        theta_u = float(action[1])
        alphas = action[2:]

        # ---- Move UAV ----
        area_width, area_height = self.map_cfg['area']
        self.uav_position[0] += self.tau * v_u * np.cos(theta_u)
        self.uav_position[1] += self.tau * v_u * np.sin(theta_u)
        self.uav_position[0] = np.clip(self.uav_position[0], 0, area_width)
        self.uav_position[1] = np.clip(self.uav_position[1], 0, area_height)

        # ---- Move vehicles ----
        for n in range(self.num_vehicles):
            self.vehicle_positions[n, 0] += self.tau * self.vehicle_velocities[n] * np.cos(self.vehicle_angles[n])
            self.vehicle_positions[n, 1] += self.tau * self.vehicle_velocities[n] * np.sin(self.vehicle_angles[n])
            self.vehicle_positions[n, 0] = np.clip(self.vehicle_positions[n, 0], 0, area_width)
            self.vehicle_positions[n, 1] = np.clip(self.vehicle_positions[n, 1], 0, area_height)

        # ---- NOMA Step 1: Update small-scale fading (eq. 7) ----
        self._update_small_scale_fading()

        # ---- Decode associations and determine offload powers ----
        associations = np.zeros((self.num_vehicles, self.num_ens), dtype=int)
        offload_ratios = np.zeros(self.num_vehicles)
        offload_powers = np.zeros(self.num_vehicles)

        for n in range(self.num_vehicles):
            alpha_n = float(alphas[n])
            if alpha_n <= 1 / 3:
                # Local only
                offload_ratios[n] = 0.0
                offload_powers[n] = 0.0
            elif alpha_n <= 2 / 3:
                # Offload to nearest RSU
                distances = [np.linalg.norm(self.vehicle_positions[n] - rp)
                             for rp in self.rsu_positions]
                closest_rsu = int(np.argmin(distances))
                associations[n, closest_rsu] = 1
                offload_ratios[n] = 3 * alpha_n - 1
                offload_powers[n] = offload_ratios[n] * self.p_max_offload
            else:
                # Offload to UAV
                associations[n, self.num_rsus] = 1
                offload_ratios[n] = 1.0
                offload_powers[n] = self.p_max_offload

        # ---- NOMA Step 2: Compute ZF SINR using CURRENT channel + offload powers ----
        sinr_rsu_curr, sinr_uav_curr = self._compute_noma_sinr(offload_powers)

        # ---- Process each vehicle ----
        en_associations = associations.sum(axis=0)
        total_cost = 0.0
        penalty = 0.0

        for n in range(self.num_vehicles):
            task = self.current_tasks[n]
            if task is None:
                # No task: still update buffer with zero arrivals
                self._update_vehicle_buffer(n, 0, 0, 0)
                continue

            data_size, max_latency = task
            en_idx = int(np.argmax(associations[n])) if associations[n].sum() > 0 else -1
            offload_ratio = offload_ratios[n]
            local_power_fraction = 1 - offload_ratio  # heuristic: inverse of offload

            # --- Local processing (eq. 16-17) ---
            local_power = local_power_fraction * self.p_max_offload
            d_local, f_m = self._compute_local_processed_bits(n, local_power)

            # --- Offload processing via NOMA (eq. 18) ---
            d_offload = 0.0
            tau_trans = 0.0
            if en_idx >= 0 and offload_ratio > 0:
                # Rate uses SINR from PREVIOUS slot (as VU only gets SINR from BS at next slot)
                rate, sinr_used = self._compute_noma_transmission_rate(n, en_idx, offload_powers[n])
                # Bits offloaded = min(rate_capacity, actual offload portion)
                offload_bits_target = offload_ratio * data_size
                d_offload = min(rate, offload_bits_target)
                tau_trans = (d_offload / (rate / self.tau)) if rate > 0 else 0
            else:
                offload_bits_target = 0.0

            # --- Buffer update (eq. 15) ---
            self._update_vehicle_buffer(n, d_offload, d_local, data_size)

            # --- Latency computation (eqs. 10-17 base env, adapted) ---
            # Local latency
            X_c = self.net_cfg.get('cpu_cycles_per_bit', 500)
            F_n = self.net_cfg.get('vehicle_compute_capacity', 2.15e9)
            Q_n = self.vehicle_queues[n]
            tau_local = X_c * Q_n / F_n + (1 - offload_ratio) * X_c * data_size / F_n

            # Offload latency (transmission + EN queue + EN compute)
            tau_offload = 0.0
            if en_idx >= 0 and offload_ratio > 0:
                if en_idx < self.num_rsus:
                    F_m = self.net_cfg.get('rsu_compute_capacity', 10e9)
                else:
                    F_m = self.net_cfg.get('uav_compute_capacity', 5e9)
                Q_m = self.en_queues[en_idx]
                tau_offload = tau_trans + X_c * Q_m / F_m + offload_ratio * X_c * data_size / F_m

            tau_total = max(tau_local, tau_offload)

            # --- AoI cost (eq. 24) ---
            aoi_cost = tau_total + self.task_intervals[n]

            # --- Energy cost ---
            energy_local = self._compute_local_energy(n, local_power, f_m)
            energy_offload = self._compute_offload_energy(n, offload_powers[n], tau_trans)
            energy_total = energy_local + energy_offload

            # --- Rental price (eq. 26) ---
            rental_price = 0.0
            if en_idx >= 0 and offload_ratio > 0:
                rental_price = self._compute_rental_price(n, en_idx, offload_bits_target)

            # --- Weighted vehicle cost (eq. 27) ---
            gamma_A = self.net_cfg.get('gamma_A', 1.0)
            gamma_E = self.net_cfg.get('gamma_E', 1.0)
            gamma_P = self.net_cfg.get('gamma_P', 1.0)
            vehicle_cost = gamma_A * aoi_cost + gamma_E * energy_total + gamma_P * rental_price
            total_cost += vehicle_cost

            # --- Constraint penalties ---
            if tau_total > max_latency:
                penalty += self.net_cfg.get('c2', 10.0)

            # --- Update legacy queues for EN congestion tracking ---
            D_comp_n = min(self.tau * F_n / X_c, self.vehicle_queues[n])
            self.vehicle_queues[n] = max(self.vehicle_queues[n] - D_comp_n, 0) + (1 - offload_ratio) * data_size

            if en_idx >= 0 and offload_ratio > 0:
                if en_idx < self.num_rsus:
                    F_m = self.net_cfg.get('rsu_compute_capacity', 10e9)
                else:
                    F_m = self.net_cfg.get('uav_compute_capacity', 5e9)
                D_comp_m = min(self.tau * F_m / X_c, self.en_queues[en_idx])
                self.en_queues[en_idx] = max(self.en_queues[en_idx] - D_comp_m, 0) + offload_ratio * data_size

        # ---- Association constraint penalties (eq. 28f-28g) ----
        for m in range(self.num_rsus):
            if en_associations[m] > self.net_cfg.get('rsu_max_associations', self.num_vehicles):
                penalty += self.net_cfg.get('c1', 5.0)
        if en_associations[self.num_rsus] > self.net_cfg.get('uav_max_associations', self.num_vehicles):
            penalty += self.net_cfg.get('c1', 5.0)

        # ---- UAV energy (eq. 20) ----
        uav_energy_consumed = self._compute_uav_energy(v_u)
        self.uav_residual_energy -= uav_energy_consumed
        if self.uav_residual_energy < 0:
            penalty += self.net_cfg.get('c3', 20.0)

        # ---- NOMA Step 3: Store current SINR as previous for next slot ----
        # (VU m observes gamma_m(t-1) at slot t — eq. 19 of Zhu et al.)
        self._sinr_prev_rsu = sinr_rsu_curr.copy()
        self._sinr_prev_uav = sinr_uav_curr.copy()

        # ---- Reward (eq. 33) ----
        reward = -(total_cost + penalty)

        # ---- Advance time ----
        self.task_intervals += self.tau
        self.current_tasks = self._generate_tasks()
        self.current_step += 1
        done = self.current_step >= self.td3_cfg['max_steps']

        next_state = self._get_state()

        info = {
            'total_cost': total_cost,
            'penalty': penalty,
            'uav_energy': self.uav_residual_energy,
            'sinr_rsu_mean': float(np.mean(self._sinr_prev_rsu)),
            'sinr_uav_mean': float(np.mean(self._sinr_prev_uav)),
            'buffer_mean': float(np.mean(self.vehicle_buffers)),
        }
        return next_state, reward, done, info



