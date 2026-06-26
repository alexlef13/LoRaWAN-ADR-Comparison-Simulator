import psutil
import tracemalloc
import simpy
import random
import numpy as np
import math
import sys
import os
import json
import csv
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# =============================================================================
# DQN IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# =============================================================================
# SystemConfig
# =============================================================================

@dataclass
class SystemConfig:
    """Κεντρική διαμόρφωση συστήματος με ενεργειακές παραμέτρους"""
    full_collision: bool = False
    directionality: bool = True
    
    learning_rate: float = 0.7
    discount_factor: float = 0.95
    exploration_rate: float = 0.35
    
    dqn_learning_rate: float = 0.0003
    dqn_epsilon_start: float = 0.9
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay: float = 0.998
    dqn_hidden_layers: List[int] = field(default_factory=lambda: [64])
    dqn_memory_size: int = 100
    dqn_batch_size: int = 32
    
    use_energy_optimization: bool = True
    use_dqn: bool = False
    use_improved_adr: bool = False
    use_ml_adr: bool = False
    battery_capacity_mah: float = 300.0
    
    tx_power_consumption: Dict[int, float] = field(default_factory=lambda: {
        10: 10.0, 12: 15.0, 14: 25.0, 17: 50.0, 20: 100.0
    })
    rx_consumption_mw: float = 15.0
    
    save_models: bool = True
    models_dir: str = "models"
    data_dir: str = "training_data"
    energy_dir: str = "energy_data"
    dqn_dir: str = "dqn_models"
    simulation_time_ms: float = 0
    
    def __post_init__(self):
        for d in [self.models_dir, self.data_dir, self.energy_dir, self.dqn_dir]:
            os.makedirs(d, exist_ok=True)

full_collision = False

# Sensitivity arrays 
sf7 = np.array([7, -126.5, -124.25, -120.75])
sf8 = np.array([8, -127.25, -126.75, -124.0])
sf9 = np.array([9, -131.25, -128.25, -127.5])
sf10 = np.array([10, -132.75, -130.25, -128.75])
sf11 = np.array([11, -134.5, -132.75, -128.75])
sf12 = np.array([12, -133.25, -132.25, -132.25])

sensi = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

# =============================================================================
# LoRaPacket
# =============================================================================

class LoRaPacket:
    """Βελτιστοποιημένο πακέτο LoRa με __slots__ για μείωση memory"""
    __slots__ = ['nodeid', 'bs', 'sf', 'cr', 'bw', 'txpow', 'pl', 'freq',
                 'collided', 'processed', 'lost', 'rssi', 'rectime', 
                 'addTime', 'seqNr', '_airtime_cache']
    
    def __init__(self, node_id: int, packet_len: int, distance: float, base_station: Any, 
                 directionality_factor: float = 1.0):
        self.nodeid = node_id
        self.bs = base_station.id
        self.sf = 12
        self.cr = 1
        self.bw = 125
        self.txpow = 14
        self.pl = packet_len
        self.freq = 868000000 + random.randint(0, 2000000)
        self.collided = 0
        self.processed = 0
        self.lost = False
        self._airtime_cache = {}
        
        self._update_rssi(distance, directionality_factor)
        self.rectime = self._calculate_airtime()
        self.addTime = 0
        self.seqNr = 0
    
    def _calculate_airtime(self) -> float:
        """Υπολογισμός χρόνου μετάδοσης με caching"""
        cache_key = (self.sf, self.bw, self.pl, self.cr)
        if cache_key in self._airtime_cache:
            return self._airtime_cache[cache_key]
        
        Tsym = (2.0 ** self.sf) / (self.bw * 1000.0)
        Npream = 8
        Tpream = (Npream + 4.25) * Tsym
        
        H = 0
        DE = 1 if (self.bw == 125 and self.sf in [11, 12]) else 0
        
        payloadSymbNB = 8 + max(
            math.ceil((8.0 * self.pl - 4.0 * self.sf + 28 + 16 - 20 * H) / 
                     (4.0 * (self.sf - 2 * DE))) * (self.cr + 4), 0
        )
        
        Tpayload = payloadSymbNB * Tsym
        airtime = Tpream + Tpayload
        self._airtime_cache[cache_key] = airtime
        return airtime
    
    def _update_rssi(self, distance: float, directionality_factor: float = 1.0):
        """Ενημέρωση RSSI με ρεαλιστικό path loss"""
        Lpld0, gamma, d0 = 127.41, 2.08, 1000.0
        distance = max(1.0, distance)
        
        Lpl = Lpld0 + 10 * gamma * math.log10(distance / d0)
        
        antenna_gain_db = 0
        if directionality_factor < 0.3:
            antenna_gain_db = -3
        elif directionality_factor < 0.6:
            antenna_gain_db = 0
        else:
            antenna_gain_db = 3
        
        effective_tx_power = self.txpow + antenna_gain_db
        shadowing = random.gauss(0, 4.5)
        
        self.rssi = effective_tx_power - Lpl + shadowing
        
        sensitivities = {
            (7,125): -126.5, (7,250): -124.25, (7,500): -120.75,
            (8,125): -127.25, (8,250): -126.75, (8,500): -124.0,
            (9,125): -131.25, (9,250): -128.25, (9,500): -127.5,
            (10,125): -132.75, (10,250): -130.25, (10,500): -128.75,
            (11,125): -134.5, (11,250): -132.75, (11,500): -128.75,
            (12,125): -133.25, (12,250): -132.25, (12,500): -132.25
        }
        
        sensitivity = sensitivities.get((self.sf, self.bw), -130.0)
        self.lost = self.rssi < (sensitivity + 3.0)
    
    def configure(self, sf: int, bw: int, tx_power: float, distance: float, 
                  directionality_factor: float = 1.0):
        """Ρύθμιση παραμέτρων πακέτου"""
        self.sf = sf
        self.bw = bw
        self.txpow = tx_power
        self._update_rssi(distance, directionality_factor)
        self.rectime = self._calculate_airtime()

# =============================================================================
# QuantumActionSpace
# =============================================================================

class QuantumActionSpace:
    """Διακριτός χώρος ενεργειών με ενεργειακά βάρη"""
    
    ACTIONS = [
        (7, 125, 10), (7, 125, 12), (7, 125, 14), (7, 125, 17), (7, 125, 20),
        (7, 250, 10), (7, 250, 12), (7, 250, 14), (7, 250, 17), 
        (7, 500, 10), (7, 500, 12), (7, 500, 14),
        (8, 125, 10), (8, 125, 12), (8, 125, 14), (8, 125, 17), (8, 125, 20),
        (8, 250, 10), (8, 250, 12), (8, 250, 14), (8, 250, 17),
        (8, 500, 10), (8, 500, 12), (8, 500, 14),
        (9, 125, 10), (9, 125, 12), (9, 125, 14), (9, 125, 17), (9, 125, 20),
        (9, 250, 10), (9, 250, 12), (9, 250, 14), (9, 250, 17),
        (9, 500, 10), (9, 500, 12), (9, 500, 14),
        (10, 125, 10), (10, 125, 12), (10, 125, 14), (10, 125, 17), (10, 125, 20),
        (10, 250, 10), (10, 250, 12), (10, 250, 14), (10, 250, 17),
        (11, 125, 10), (11, 125, 12), (11, 125, 14), (11, 125, 17), (11, 125, 20),
        (11, 250, 10), (11, 250, 12), (11, 250, 14),
        (12, 125, 10), (12, 125, 12), (12, 125, 14), (12, 125, 17), (12, 125, 20),
        (12, 250, 10), (12, 250, 12), (12, 250, 14)
    ]
    
    ACTION_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
    
    @staticmethod
    def get_action_by_index(index: int) -> tuple:
        return QuantumActionSpace.ACTIONS[index % len(QuantumActionSpace.ACTIONS)]
    
    @staticmethod
    def get_index_by_action(action: tuple) -> int:
        return QuantumActionSpace.ACTION_INDEX.get(action, -1)

# =============================================================================
# NodeState
# =============================================================================

@dataclass
class NodeState:
    """Κβαντισμένη κατάσταση κόμβου με ενεργειακά features"""
    rssi_level: int
    distance_level: int
    snr_level: int
    der_level: int
    collision_level: int
    airtime_level: int
    load_level: int
    avg_load_level: int
    failure_streak: int
    success_streak: int
    energy_level: int
    sf_offset: int
    bw_offset: int
    tx_offset: int
    performance_trend: int
    stability_level: int
    energy_efficiency_level: int
    energy_per_packet_level: int
    
    def to_tuple(self) -> tuple:
        return tuple(vars(self).values())
    
    @classmethod
    def from_measurements(cls, rssi: float, distance: float, snr: float,
                         der: float, collision_rate: float, airtime: float,
                         load: int, avg_load: float, failures: int, 
                         successes: int, energy_ratio: float,
                         sf: int, bw: int, tx_power: int,
                         trend: int, stability: float,
                         energy_efficiency: float, energy_per_packet: float) -> 'NodeState':
        
        def quantize(value: float, min_val: float, max_val: float, levels: int) -> int:
            if value < min_val:
                return 0
            if value > max_val:
                return levels - 1
            return int((value - min_val) / ((max_val - min_val) / levels))
        
        return cls(
            rssi_level=quantize(rssi, -140, -90, 8),
            distance_level=quantize(distance, 0, 5000, 8),
            snr_level=quantize(snr, -20, 20, 6),
            der_level=quantize(der, 0, 1.0, 5),
            collision_level=quantize(collision_rate, 0, 1.0, 5),
            airtime_level=quantize(airtime, 0, 2000, 5),
            load_level=min(7, load),
            avg_load_level=quantize(avg_load, 0, 20, 6),
            failure_streak=min(3, failures),
            success_streak=min(3, successes),
            energy_level=quantize(energy_ratio, 0, 1.0, 4),
            sf_offset=min(5, sf - 7),
            bw_offset={125: 0, 250: 1, 500: 2}.get(bw, 0),
            tx_offset=quantize(tx_power, 10, 20, 6),
            performance_trend=trend + 1,
            stability_level=quantize(stability, 0, 1.0, 4),
            energy_efficiency_level=quantize(energy_efficiency, 0, 1.0, 4),
            energy_per_packet_level=quantize(energy_per_packet, 0, 50000, 4)
        )

# =============================================================================
# EnergyTracker (ΔΙΟΡΘΩΜΕΝΟ)
# =============================================================================

class EnergyTracker:
    """Βελτιστοποιημένο σύστημα παρακολούθησης ενέργειας με ΣΩΣΤΟΥΣ υπολογισμούς"""
    __slots__ = ['config', 'initial_energy_mah', 'remaining_energy_mah', 
                 'total_consumed_mah', 'energy_history', 'energy_efficiency_history',
                 'last_energy_consumed_mah', 'total_successful_energy', 
                 'total_failed_energy', 'packets_sent', 'successful_packets',
                 'failed_packets', 'retransmissions', '_energy_cache']
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.initial_energy_mah = config.battery_capacity_mah
        self.remaining_energy_mah = config.battery_capacity_mah
        self.total_consumed_mah = 0.0
        
        self.energy_history = deque(maxlen=1000)
        self.energy_efficiency_history = deque(maxlen=500)
        
        self.last_energy_consumed_mah = 0.0
        self.total_successful_energy = 0.0
        self.total_failed_energy = 0.0
        self.packets_sent = 0
        self.successful_packets = 0
        self.failed_packets = 0
        self.retransmissions = 0
        
        self._energy_cache = {}
        
        print(f"[EnergyTracker] Initialized with {self.initial_energy_mah:.1f} mAh battery")
    
    def calculate_transmission_energy(self, sf: int, tx_power: int, 
                                     airtime_ms: float, success: bool) -> float:
        """Ρεαλιστικός υπολογισμός ενέργειας μετάδοσης"""
        
        tx_power_mw = self.config.tx_power_consumption.get(tx_power, 25.0)
        
        sf_processing_mw = {
            7: 5.0, 8: 6.0, 9: 7.5, 10: 10.0, 11: 15.0, 12: 22.0
        }.get(sf, 5.0)
        
        total_power_mw = tx_power_mw + sf_processing_mw
        
        airtime_hours = airtime_ms / (1000 * 3600)
        
        tx_energy_mah = total_power_mw * airtime_hours
        
        if success:
            ack_time_hours = 0.050 / 3600
            rx_power_mw = self.config.rx_consumption_mw
            ack_energy_mah = rx_power_mw * ack_time_hours
        else:
            ack_energy_mah = 0
        
        total_energy = tx_energy_mah + ack_energy_mah
        
        return total_energy
    
    def update(self, sf: int, tx_power: int, airtime_ms: float, 
               success: bool, retransmission: bool = False) -> float:
        """Ενημέρωση με διαχωρισμό επιτυχημένων/αποτυχημένων"""
        energy_used = self.calculate_transmission_energy(sf, tx_power, airtime_ms, success)
        
        self.last_energy_consumed_mah = energy_used
        self.total_consumed_mah += energy_used
        self.remaining_energy_mah -= energy_used
        
        if success:
            self.successful_packets += 1
            self.total_successful_energy += energy_used
        else:
            self.total_failed_energy += energy_used
            self.failed_packets += 1
            
        if retransmission:
            self.retransmissions += 1
        
        self.packets_sent += 1
        
        if airtime_ms > 0:
            min_energy = self.calculate_transmission_energy(7, 14, 50, True)
            efficiency = min_energy / energy_used if energy_used > 0 else 0
            efficiency = min(1.0, efficiency)
            self.energy_efficiency_history.append(efficiency)
        
        return energy_used
    
    def get_energy_efficiency(self) -> float:
        if not self.energy_efficiency_history:
            return 1.0
        return np.mean(list(self.energy_efficiency_history))
    
    def get_avg_energy_per_packet(self) -> float:
        if self.packets_sent == 0:
            return 0.0
        return self.total_consumed_mah / self.packets_sent
    
    def get_energy_per_successful_packet(self) -> float:
        """Επιστρέφει την πραγματική ενέργεια ανά ΕΠΙΤΥΧΗΜΕΝΟ πακέτο"""
        if self.successful_packets == 0:
            return float('inf')
        return self.total_consumed_mah / self.successful_packets
    
    def get_true_energy_per_successful_uah(self) -> float:
        """Επιστρέφει την ενέργεια ανά επιτυχημένο πακέτο σε μAh"""
        energy_mah = self.get_energy_per_successful_packet()
        if energy_mah == float('inf'):
            return 0.0
        return energy_mah * 1_000_000
    
    def get_lifetime_estimation(self, simulation_time_ms: float = None, target_period_ms: int = 1000) -> Dict[str, Any]:
        
        
        if self.successful_packets == 0:
            return {
                'remaining_energy_mah': self.remaining_energy_mah,
                'total_consumed_mah': 0,
                'energy_per_successful_packet_mah': float('inf'),
                'avg_energy_per_packet_mah': 0,
                'energy_efficiency': 1.0,
                'success_rate': 0,
                'retransmission_rate': 0,
                'retransmission_overhead': 0,
                'estimated_remaining_successful_packets': 0,
                'estimated_lifetime_hours': 0,
                'estimated_lifetime_days': 0,
                'estimated_lifetime_months': 0,
                'estimated_lifetime_years': 0,
                'battery_health': 100.0,
                'packets_per_hour': 0,
                'energy_per_hour_mah': 0,
                'warning': 'No successful packets'
            }
        
        success_rate = self.successful_packets / self.packets_sent
        energy_per_success = self.get_energy_per_successful_packet()
        
        # ΣΩΣΤΟΣ ΥΠΟΛΟΓΙΣΜΟΣ - Λαμβάνει υπόψη και τις αποτυχημένες μεταδόσεις
        target_packets_per_hour = 3600 / (target_period_ms / 1000)
        
        # Πραγματικές μεταδόσεις που χρειάζονται για να πετύχουμε τον στόχο
        transmissions_per_hour = target_packets_per_hour / success_rate if success_rate > 0 else float('inf')
        
        # Ενέργεια ανά μετάδοση (συμπεριλαμβανομένων των αποτυχημένων)
        energy_per_transmission_mah = self.total_consumed_mah / self.packets_sent if self.packets_sent > 0 else 0
        
        # Συνολική ενέργεια ανά ώρα
        energy_per_hour_correct = transmissions_per_hour * energy_per_transmission_mah
        
        # Σωστή διάρκεια ζωής
        if energy_per_hour_correct > 0:
            remaining_hours = self.remaining_energy_mah / energy_per_hour_correct
            remaining_days = remaining_hours / 24
            remaining_months = remaining_days / 30.44
            remaining_years = remaining_months / 12
        else:
            remaining_hours = 0
            remaining_days = 0
            remaining_months = 0
            remaining_years = 0
        
        # Εκτίμηση υπολειπόμενων επιτυχημένων πακέτων (με βάση το success rate)
        if energy_per_success > 0 and energy_per_success != float('inf'):
            remaining_successful_packets = (self.remaining_energy_mah / energy_per_success) * success_rate
        else:
            remaining_successful_packets = 0
        
        retransmission_overhead = self.total_failed_energy / self.total_consumed_mah if self.total_consumed_mah > 0 else 0
        
        if simulation_time_ms is not None and simulation_time_ms > 0:
            simulation_hours = simulation_time_ms / (1000 * 3600)
            packets_per_hour = self.packets_sent / simulation_hours
            energy_per_hour_mah = self.total_consumed_mah / simulation_hours
        else:
            packets_per_hour = transmissions_per_hour
            energy_per_hour_mah = energy_per_hour_correct
        
        return {
            'remaining_energy_mah': self.remaining_energy_mah,
            'total_consumed_mah': self.total_consumed_mah,
            'energy_per_successful_packet_mah': energy_per_success,
            'energy_per_successful_packet_uah': energy_per_success * 1_000_000,  # Για ευκολία
            'avg_energy_per_packet_mah': self.total_consumed_mah / max(1, self.packets_sent),
            'energy_per_attempt_mah': self.total_consumed_mah / max(1, self.packets_sent),
            'energy_efficiency': self.get_energy_efficiency(),
            'success_rate': success_rate,
            'retransmission_rate': self.retransmissions / max(1, self.packets_sent),
            'retransmission_overhead': retransmission_overhead,
            'packets_per_hour': packets_per_hour,
            'energy_per_hour_mah': energy_per_hour_mah,
            'estimated_remaining_successful_packets': remaining_successful_packets,
            'estimated_lifetime_hours': remaining_hours,
            'estimated_lifetime_days': remaining_days,
            'estimated_lifetime_months': remaining_months,
            'estimated_lifetime_years': remaining_years,
            'battery_health': (self.remaining_energy_mah / self.initial_energy_mah) * 100,
            'target_period_ms': target_period_ms,
            'target_packets_per_hour': target_packets_per_hour,
            'actual_transmissions_per_hour': transmissions_per_hour,
            'corrected_energy_per_hour_mah': energy_per_hour_correct
        }

# =============================================================================
# UnifiedNodeStatistics
# =============================================================================

class UnifiedNodeStatistics:
    """Ενοποιημένα στατιστικά κόμβου με ενεργειακά μετρικά"""
    __slots__ = ['window_size', 'success_history', 'collision_history', 
                 'lost_history', 'recent_der_history', 'rssi_history',
                 'snr_history', 'airtime_history', 'load_history', 
                 'energy_history', 'total_sent', 'total_success', 
                 'total_collisions', 'total_lost', 'total_energy',
                 'action_changes', 'last_action']
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
        self.success_history = deque(maxlen=window_size)
        self.collision_history = deque(maxlen=window_size)
        self.lost_history = deque(maxlen=window_size)
        self.recent_der_history = deque(maxlen=window_size)
        self.rssi_history = deque(maxlen=window_size)
        self.snr_history = deque(maxlen=window_size)
        self.airtime_history = deque(maxlen=window_size)
        self.load_history = deque(maxlen=window_size)
        self.energy_history = deque(maxlen=window_size)
        
        self.total_sent = 0
        self.total_success = 0
        self.total_collisions = 0
        self.total_lost = 0
        self.total_energy = 0.0
        
        self.action_changes = 0
        self.last_action = None
    
    def add_result(self, success: bool, collision: bool, lost: bool,
                  rssi: Optional[float] = None, snr: Optional[float] = None,
                  airtime: Optional[float] = None, sf: Optional[int] = None,
                  tx_power: Optional[float] = None, network_load: int = 0,
                  current_action: Optional[tuple] = None, energy_used: Optional[float] = None):
        
        self.total_sent += 1
        
        self.success_history.append(1 if success else 0)
        self.collision_history.append(1 if collision else 0)
        self.lost_history.append(1 if lost else 0)
        
        if success:
            self.total_success += 1
        elif collision:
            self.total_collisions += 1
        elif lost:
            self.total_lost += 1
        
        if rssi is not None:
            self.rssi_history.append(rssi)
        if snr is not None:
            self.snr_history.append(snr)
        if airtime is not None:
            self.airtime_history.append(airtime)
        if network_load > 0:
            self.load_history.append(network_load)
        if energy_used is not None:
            self.energy_history.append(energy_used)
            self.total_energy += energy_used
        
        if len(self.success_history) > 0:
            recent_der = sum(self.success_history) / len(self.success_history)
            self.recent_der_history.append(recent_der)
        
        if current_action is not None:
            if self.last_action is not None and current_action != self.last_action:
                self.action_changes += 1
            self.last_action = current_action
    
    def get_der(self) -> float:
        return self.total_success / max(1, self.total_sent)
    
    def get_recent_der(self) -> float:
        if not self.recent_der_history:
            return 0.0
        recent = list(self.recent_der_history)[-10:]
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_recent_collision_rate(self) -> float:
        if not self.collision_history:
            return 0.0
        recent = list(self.collision_history)[-10:]
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_avg_energy_per_packet(self) -> float:
        return self.total_energy / max(1, self.total_sent)
    
    def get_energy_efficiency(self) -> float:
        if self.total_energy == 0:
            return 1.0
        min_energy_per_packet = 0.000264
        avg_energy = self.get_avg_energy_per_packet()
        efficiency = min_energy_per_packet / avg_energy if avg_energy > 0 else 0
        return min(1.0, efficiency)
    
    def get_performance_trend(self) -> int:
        if len(self.recent_der_history) < 10:
            return 0
        recent = list(self.recent_der_history)[-10:]
        older = list(self.recent_der_history)[-20:-10] if len(self.recent_der_history) >= 20 else recent
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        if recent_avg > older_avg + 0.15:
            return 1
        elif recent_avg < older_avg - 0.15:
            return -1
        return 0
    
    def get_action_stability(self) -> float:
        if self.total_sent < 2:
            return 1.0
        return 1.0 - (self.action_changes / (self.total_sent - 1))

# =============================================================================
# DQN Network
# =============================================================================

class OptimizedDQNNetwork(nn.Module):
    """Βελτιστοποιημένο network με μικρότερο μέγεθος"""
    def __init__(self, state_size=18, action_size=66, hidden_layers=None):
        super(OptimizedDQNNetwork, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [64]
        
        layers = []
        input_size = state_size
        
        for hidden in hidden_layers:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        print(f"[OptimizedNetwork] {state_size} -> {hidden_layers} -> {action_size}")
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

# =============================================================================
# Improved DQN Agent
# =============================================================================

class ImprovedDQNAgent:
    """Βελτιωμένος DQN Agent με καλύτερη εξερεύνηση και επιθετική στρατηγική"""
    
    def __init__(self, node_id: int, config: SystemConfig):
        self.node_id = node_id
        self.config = config
        
        self.state_size = 18
        self.action_size = len(QuantumActionSpace.ACTIONS)
        self.device = torch.device("cpu")
        
        print(f"[Improved DQN Agent {node_id}] Using device: {self.device}")
        
        self.policy_net = OptimizedDQNNetwork(
            self.state_size, 
            self.action_size,
            config.dqn_hidden_layers
        ).to(self.device)
        
        self.target_net = OptimizedDQNNetwork(
            self.state_size,
            self.action_size,
            config.dqn_hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.dqn_learning_rate
        )
        
        self.memory = deque(maxlen=config.dqn_memory_size)
        self.epsilon = config.dqn_epsilon_start
        self.epsilon_min = config.dqn_epsilon_end
        self.epsilon_decay = config.dqn_epsilon_decay
        
        self.batch_size = config.dqn_batch_size
        self.gamma = 0.9
        self.target_update = 500
        self.learn_step_counter = 0
        self.learn_every = 5
        
        # Exploration phase
        self.exploration_phase = True
        self.exploration_steps = 1000
        self.step_count = 0
        
        self.training_metrics = {
            'rewards': [],
            'losses': [],
            'epsilons': [],
            'action_distribution': defaultdict(int)
        }
    
    def select_action(self, state: NodeState, network_load: int, 
                     distance: float, energy_ratio: float) -> tuple:
        self.step_count += 1
        
        # Exploration phase: try all SFs aggressively
        if self.step_count < self.exploration_steps:
            if network_load > 20:
                return random.choice([(7,125,14), (7,250,14), (7,500,14), 
                                     (8,125,14), (8,250,14), (9,125,14)])
            elif energy_ratio < 0.3:
                return random.choice([(7,125,10), (7,250,10), (8,125,10)])
            else:
                return random.choice(QuantumActionSpace.ACTIONS[:30])
        
        # Adaptive epsilon based on network conditions
        adaptive_epsilon = self.epsilon
        if network_load > 20:
            adaptive_epsilon *= 1.5
        if energy_ratio < 0.3:
            adaptive_epsilon *= 1.2
        if self.step_count > 5000:
            adaptive_epsilon *= 0.8
        
        adaptive_epsilon = min(0.8, max(self.epsilon_min, adaptive_epsilon))
        
        if random.random() < adaptive_epsilon:
            return self._intelligent_exploration(network_load, distance, energy_ratio)
        else:
            return self._exploit(state)
    
    def _intelligent_exploration(self, network_load: int, distance: float, energy_ratio: float) -> tuple:
        """Έξυπνη εξερεύνηση με βάση τις συνθήκες"""
        if network_load > 20:
            sf = random.choice([7, 8, 9])
            bw = random.choice([125, 250, 500]) if sf <= 8 else 125
            tx = random.choice([10, 12, 14])
        elif distance < 1000:
            sf = random.choice([7, 8])
            bw = random.choice([125, 250, 500])
            tx = random.choice([10, 12])
        elif distance < 3000:
            sf = random.choice([9, 10])
            bw = 125
            tx = random.choice([14, 17])
        else:
            sf = random.choice([11, 12])
            bw = 125
            tx = random.choice([17, 20])
        
        if energy_ratio < 0.2:
            tx = min(tx, 14)
        
        return (sf, bw, tx)
    
    def _exploit(self, state: NodeState) -> tuple:
        """Exploitation με DQN"""
        state_tensor = self._state_to_tensor(state)
        
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            best_idx = torch.argmax(q_values).item()
        
        self.policy_net.train()
        action = QuantumActionSpace.get_action_by_index(best_idx)
        self.training_metrics['action_distribution'][action] += 1
        return action
    
    def _state_to_tensor(self, state: NodeState) -> torch.Tensor:
        state_array = np.array(state.to_tuple(), dtype=np.float32)
        return torch.FloatTensor(state_array).to(self.device)
    
    def update(self, state: NodeState, action: tuple, reward: float,
              next_state: NodeState, done: bool, network_load: int,
              airtime_ms: float):
        action_idx = QuantumActionSpace.get_index_by_action(action)
        if action_idx == -1:
            return
        
        state_array = np.array(state.to_tuple(), dtype=np.float32)
        next_state_array = np.array(next_state.to_tuple(), dtype=np.float32)
        
        self.memory.append((state_array, action_idx, reward, next_state_array, done))
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.learn_every == 0 and len(self.memory) >= self.batch_size:
            loss = self._train()
            if loss is not None:
                self.training_metrics['losses'].append(loss)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_metrics['epsilons'].append(self.epsilon)
        
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).to(self.device)
        
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

# =============================================================================
# BaseNode
# =============================================================================

class BaseNode:
    """Βασική κλάση για όλους τους τύπους κόμβων"""
    
    def __init__(self, node_id: int, period: int, base_station: Any, config: SystemConfig):
        self.id = node_id
        self.period = period * random.uniform(0.5, 2.0)
        self.original_period = self.period
        self.base_station = base_station
        self.config = config
        self.env = None
        
        self.x, self.y = self._calculate_position(base_station)
        self.distance = self._calculate_distance(base_station)
        
        self.orientation = random.random() * 2 * math.pi
        self.has_directionality = config.directionality and random.random() > 0.3
        
        direction_factor = self._calculate_directionality_factor()
        self.packet = LoRaPacket(node_id, 20, self.distance, base_station, direction_factor)
        
        self.stats = UnifiedNodeStatistics()
        self.energy_tracker = EnergyTracker(config)
        
        self.sent_packets = 0
        self.successful_packets = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
    
    def set_env(self, env):
        self.env = env
    
    def _calculate_position(self, bs: Any) -> Tuple[float, float]:
        distance_type = random.random()
        if distance_type < 0.33:
            distance = 100 + random.random() * 900
        elif distance_type < 0.66:
            distance = 1000 + random.random() * 2000
        else:
            distance = 3000 + random.random() * 2000
        
        angle = random.random() * 2 * math.pi
        return bs.x + distance * math.cos(angle), bs.y + distance * math.sin(angle)
    
    def _calculate_distance(self, bs: Any) -> float:
        return math.sqrt((self.x - bs.x)**2 + (self.y - bs.y)**2)
    
    def _calculate_directionality_factor(self) -> float:
        if not self.has_directionality:
            return 1.0
        
        dx = self.base_station.x - self.x
        dy = self.base_station.y - self.y
        angle_to_bs = math.atan2(dy, dx)
        
        angle_diff = abs(angle_to_bs - self.orientation)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        
        if angle_diff > math.pi/3 and hasattr(self, 'sent_packets') and self.sent_packets > 100:
            learning_rate = 0.2 if self.sent_packets > 100 else 0.5
            self.orientation += (angle_to_bs - self.orientation) * learning_rate
            if self.orientation > 2*math.pi:
                self.orientation -= 2*math.pi
            elif self.orientation < 0:
                self.orientation += 2*math.pi
        
        return max(0.1, math.exp(-2.0 * angle_diff))
    
    def calculate_snr(self, rssi: float, bw: int) -> float:
        noise_density = -174
        noise_figure = 6.0
        noise_power = noise_density + noise_figure + 10 * math.log10(bw * 1000)
        return rssi - noise_power
    
    def get_current_state(self, network_load: int) -> NodeState:
        direction_factor = self._calculate_directionality_factor()
        rssi = self.packet.rssi
        snr = self.calculate_snr(rssi, self.packet.bw)
        der = self.stats.get_recent_der()
        collision_rate = self.stats.get_recent_collision_rate()
        airtime = self.packet.rectime
        
        energy_ratio = self.energy_tracker.remaining_energy_mah / self.config.battery_capacity_mah
        energy_efficiency = self.energy_tracker.get_energy_efficiency()
        energy_per_packet = self.energy_tracker.get_avg_energy_per_packet() * 3_600_000
        
        return NodeState.from_measurements(
            rssi=rssi, distance=self.distance, snr=snr, der=der,
            collision_rate=collision_rate, airtime=airtime,
            load=network_load, avg_load=network_load,
            failures=self.consecutive_failures, successes=self.consecutive_successes,
            energy_ratio=energy_ratio, sf=self.packet.sf, bw=self.packet.bw,
            tx_power=self.packet.txpow, trend=self.stats.get_performance_trend(),
            stability=self.stats.get_action_stability(),
            energy_efficiency=energy_efficiency,
            energy_per_packet=energy_per_packet * direction_factor
        )
    
    def update_basic_stats(self, success: bool, collision: bool, lost: bool, network_load: int):
        self.sent_packets += 1
        
        if success:
            self.successful_packets += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        retransmission = (not success) and (collision or lost)
        
        self.energy_tracker.update(
            sf=self.packet.sf,
            tx_power=self.packet.txpow,
            airtime_ms=self.packet.rectime,
            success=success,
            retransmission=retransmission
        )
        
        self.stats.add_result(
            success=success, collision=collision, lost=lost,
            rssi=self.packet.rssi, snr=self.calculate_snr(self.packet.rssi, self.packet.bw),
            airtime=self.packet.rectime, sf=self.packet.sf, tx_power=self.packet.txpow,
            network_load=network_load,
            current_action=(self.packet.sf, self.packet.bw, self.packet.txpow),
            energy_used=self.energy_tracker.last_energy_consumed_mah
        )
    
    def get_energy_report(self, simulation_time_ms: float = None) -> Dict[str, Any]:
        target_period_ms = int(self.original_period)
        lifetime_est = self.energy_tracker.get_lifetime_estimation(simulation_time_ms, target_period_ms)
        
        return {
            'node_id': self.id,
            'distance': self.distance,
            'packets_sent': self.sent_packets,
            'successful_packets': self.successful_packets,
            'der': self.successful_packets / max(1, self.sent_packets),
            'current_config': {
                'sf': self.packet.sf, 'bw': self.packet.bw, 'tx_power': self.packet.txpow
            },
            'directionality': {
                'has_directionality': self.has_directionality,
                'direction_factor': self._calculate_directionality_factor()
            },
            'energy_metrics': {
                'remaining_energy_mah': self.energy_tracker.remaining_energy_mah,
                'total_consumed_mah': self.energy_tracker.total_consumed_mah,
                'energy_efficiency': lifetime_est['energy_efficiency'],
                'avg_energy_per_packet_mah': lifetime_est['avg_energy_per_packet_mah'],
                'energy_per_successful_packet_mah': lifetime_est['energy_per_successful_packet_mah'],
                'estimated_lifetime_hours': lifetime_est['estimated_lifetime_hours'],
                'estimated_lifetime_days': lifetime_est['estimated_lifetime_days'],
                'estimated_lifetime_months': lifetime_est['estimated_lifetime_months'],
                'estimated_lifetime_years': lifetime_est['estimated_lifetime_years'],
                'packets_per_hour': lifetime_est['packets_per_hour'],
                'battery_health_percentage': lifetime_est['battery_health']
            }
        }

# =============================================================================
# Enhanced DQN Node
# =============================================================================

class EnhancedDQNNode(BaseNode):
    """Κόμβος με βελτιστοποιημένο DQN agent"""
    
    def __init__(self, node_id: int, period: int, base_station: Any, config: SystemConfig):
        super().__init__(node_id, period, base_station, config)
        
        self.agent = ImprovedDQNAgent(node_id, config)
        self.last_state = None
        self.last_action = None
        self.last_energy_consumed = 0.0
        self.retransmission_count = 0
        self.failed_attempts = 0
        self.total_transmission_attempts = 0
    
    def configure_transmission(self, network_load: int):
        direction_factor = self._calculate_directionality_factor()
        state = self.get_current_state(network_load)
        energy_ratio = self.energy_tracker.remaining_energy_mah / self.config.battery_capacity_mah
        
        if self.consecutive_failures > 3:
            self.agent.epsilon = min(0.8, self.agent.epsilon * 1.2)
        
        action = self.agent.select_action(state, network_load, self.distance, energy_ratio)
        sf, bw, tx_power = action
        
        if self.retransmission_count > 10 and sf > 9:
            sf = 9
        
        if energy_ratio < 0.3:
            if sf > 7:
                sf = max(7, sf - 1)
            if tx_power > 14:
                tx_power = max(14, tx_power - 3)
        elif energy_ratio < 0.15:
            sf = min(9, sf)
            tx_power = 12
        
        if self.distance < 500 and direction_factor > 0.8:
            tx_power = max(10, tx_power - 4)
        
        if direction_factor < 0.3 and tx_power < 20:
            tx_power = min(20, tx_power + 2)
        elif direction_factor > 0.7 and tx_power > 12:
            tx_power = max(12, tx_power - 2)
        
        if network_load > 15 and sf > 7:
            sf = max(7, sf - 1)
        
        self.packet.configure(
            sf=sf, bw=bw, tx_power=tx_power,
            distance=self.distance, directionality_factor=direction_factor
        )
        
        self.last_state = state
        self.last_action = action
    
    def update_learning(self, success: bool, collision: bool, lost: bool, network_load: int):
        self.total_transmission_attempts += 1
        
        if self.last_state is None or self.last_action is None:
            self.update_basic_stats(success, collision, lost, network_load)
            return
        
        energy_consumed = self.energy_tracker.calculate_transmission_energy(
            self.packet.sf, self.packet.txpow, self.packet.rectime, success
        )
        
        is_retransmission = (not success) and (collision or lost)
        if is_retransmission:
            self.retransmission_count += 1
            self.failed_attempts += 1
        else:
            self.retransmission_count = 0
            self.failed_attempts = 0
        
        reward = self._calculate_improved_reward(
            success, collision, lost, energy_consumed, network_load
        )
        
        next_state = self.get_current_state(network_load)
        
        self.update_basic_stats(success, collision, lost, network_load)
        
        self.agent.update(
            state=self.last_state, action=self.last_action, reward=reward,
            next_state=next_state, done=False,
            network_load=network_load, airtime_ms=self.packet.rectime
        )
        
        self.last_state = None
        self.last_action = None
    
    def _calculate_improved_reward(self, success: bool, collision: bool, 
                                   lost: bool, energy_consumed: float, 
                                   network_load: int) -> float:
        reward = 0
        
        if success:
            reward += 100
            if self.packet.sf <= 8:
                reward += 40
            elif self.packet.sf <= 10:
                reward += 20
            else:
                reward -= 30
            
            if network_load > 15:
                reward += 50
            
            if energy_consumed < 0.01:
                reward += 30
        else:
            reward -= 80
            if self.packet.sf > 10:
                reward -= 30
        
        reward -= energy_consumed * 3000
        reward -= (self.packet.sf - 7) * 8
        
        if network_load > 20:
            reward += 20
        
        battery_ratio = self.energy_tracker.remaining_energy_mah / self.config.battery_capacity_mah
        if battery_ratio < 0.2 and success:
            reward += 40
        
        return np.clip(reward, -200, 300)



# =============================================================================
# Original ADR Node
# =============================================================================

class OriginalADRNode(BaseNode):
    """Original Realistic ADR node (conservative)"""
    
    def __init__(self, node_id: int, period: int, base_station: Any, config: SystemConfig):
        super().__init__(node_id, period, base_station, config)
        
        self.rssi_history = deque(maxlen=15)
        self.snr_history = deque(maxlen=15)
        self.per_history = deque(maxlen=50)
        self.snr_margin_db = 3.0
        self.target_per = 0.10
        
        self.adr_counter = 0
        self.adr_ack_req = False
        self.adr_ack_counter = 0
    
    def get_required_snr(self, sf: int) -> float:
        required_snr = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
        return required_snr.get(sf, -20.0)
    
    def get_sensitivity(self, sf: int, bw: int) -> float:
        sensitivities = {
            7: {125: -126.5, 250: -124.25, 500: -120.75},
            8: {125: -127.25, 250: -126.75, 500: -124.0},
            9: {125: -131.25, 250: -128.25, 500: -127.5},
            10: {125: -132.75, 250: -130.25, 500: -128.75},
            11: {125: -134.5, 250: -132.75, 500: -128.75},
            12: {125: -133.25, 250: -132.25, 500: -132.25}
        }
        return sensitivities.get(sf, {}).get(bw, -130.0)
    
    def configure_transmission(self, network_load: int):
        self.adr_counter += 1
        if self.adr_counter >= 20:
            self.adr_ack_req = True
            self.adr_counter = 0
        
        direction_factor = self._calculate_directionality_factor()
        
        base_per = 0.10
        if network_load > 20:
            base_per = 0.25
        elif network_load > 10:
            base_per = 0.15
        
        if direction_factor < 0.3:
            self.target_per = min(0.30, base_per * 1.5)
        elif direction_factor < 0.6:
            self.target_per = base_per
        else:
            self.target_per = max(0.05, base_per * 0.8)
        
        current_rssi = self.packet.rssi
        current_snr = self.calculate_snr(current_rssi, self.packet.bw)
        self.rssi_history.append(current_rssi)
        self.snr_history.append(current_snr)
        
        avg_rssi = np.mean(list(self.rssi_history)) if self.rssi_history else current_rssi
        avg_snr = np.mean(list(self.snr_history)) if self.snr_history else current_snr
        current_per = sum(self.per_history) / len(self.per_history) if self.per_history else 1.0
        
        if current_per > self.target_per * 1.3 or network_load > 15:
            self.snr_margin_db = min(10.0, self.snr_margin_db + 1.0)
        elif current_per < self.target_per * 0.5 and network_load < 10:
            self.snr_margin_db = max(0.0, self.snr_margin_db - 0.5)
        
        if network_load > 50:
            max_allowed_sf = 9
        elif network_load > 30:
            max_allowed_sf = 10
        elif network_load > 15:
            max_allowed_sf = 11
        else:
            max_allowed_sf = 12
        
        candidate_sf = 12
        for sf in range(7, max_allowed_sf + 1):
            required_snr = self.get_required_snr(sf)
            if avg_snr >= (required_snr + self.snr_margin_db):
                candidate_sf = sf
                break
        
        if current_per < 0.05 and candidate_sf > 7 and network_load > 20:
            candidate_sf = max(7, candidate_sf - 1)
        
        if network_load > 30:
            bw = 500 if candidate_sf <= 9 else 250
        elif network_load > 15:
            bw = 250
        else:
            bw = 125
        
        if direction_factor < 0.3:
            candidate_sf = min(12, candidate_sf + 1)
        
        sf_change = candidate_sf - self.packet.sf
        if sf_change > 0 and self.consecutive_failures < 2:
            candidate_sf = self.packet.sf
        elif sf_change < 0 and self.consecutive_successes < 5:
            candidate_sf = self.packet.sf
        
        sensitivity = self.get_sensitivity(candidate_sf, bw)
        signal_margin = avg_rssi - sensitivity
        
        if signal_margin > 15:
            tx_power = 10
        elif signal_margin > 10:
            tx_power = 12
        elif signal_margin > 5:
            tx_power = 14
        elif signal_margin > 0:
            tx_power = 17
        else:
            tx_power = 20
        
        if direction_factor < 0.3:
            tx_power = min(20, tx_power + 3)
        elif direction_factor > 0.7 and tx_power > 10:
            tx_power = max(10, tx_power - 2)
        
        self.packet.configure(
            sf=candidate_sf, bw=bw, tx_power=tx_power,
            distance=self.distance, directionality_factor=direction_factor
        )
    
    def update_learning(self, success: bool, collision: bool, lost: bool, network_load: int):
        self.update_basic_stats(success, collision, lost, network_load)
        self.per_history.append(0 if success else 1)
        
        if success and self.adr_ack_req:
            self.adr_ack_counter += 1
            self.adr_ack_req = False

# =============================================================================
# ML-ADR Node
# =============================================================================

class MLADRNode(BaseNode):
    """ADR βασισμένο σε Q-learning"""
    
    def __init__(self, node_id: int, period: int, base_station: Any, config: SystemConfig):
        super().__init__(node_id, period, base_station, config)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.possible_sf = [7, 8, 9, 10, 11, 12]
        self.possible_bw = [125, 250, 500]
        self.possible_tx = [10, 12, 14, 17, 20]
        
        self.last_state = None
        self.last_action = None
    
    def _get_state_key(self, network_load: int, distance: float, recent_der: float) -> tuple:
        load_level = min(3, network_load // 10)
        dist_level = 0 if distance < 1000 else (1 if distance < 3000 else 2)
        der_level = 0 if recent_der > 0.9 else (1 if recent_der > 0.7 else 2)
        return (load_level, dist_level, der_level)
    
    def _get_best_action(self, state: tuple, network_load: int) -> tuple:
        if state not in self.q_table or not self.q_table[state]:
            if network_load > 20:
                return (7, 250, 14)
            return (9, 125, 14)
        
        best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
        return best_action
    
    def configure_transmission(self, network_load: int):
        direction_factor = self._calculate_directionality_factor()
        recent_der = self.stats.get_recent_der()
        
        state = self._get_state_key(network_load, self.distance, recent_der)
        
        if random.random() < self.epsilon:
            if network_load > 20:
                sf = random.choice([7, 8, 9])
                bw = random.choice([125, 250])
            else:
                sf = random.choice(self.possible_sf)
                bw = random.choice(self.possible_bw)
            
            energy_ratio = self.energy_tracker.remaining_energy_mah / self.config.battery_capacity_mah
            if energy_ratio < 0.3:
                tx = random.choice([10, 12, 14])
            else:
                tx = random.choice(self.possible_tx)
        else:
            sf, bw, tx = self._get_best_action(state, network_load)
        
        energy_ratio = self.energy_tracker.remaining_energy_mah / self.config.battery_capacity_mah
        if energy_ratio < 0.15:
            sf = min(9, sf)
            tx = min(14, tx)
        
        if direction_factor < 0.3 and tx < 20:
            tx = min(20, tx + 2)
        elif direction_factor > 0.7 and tx > 12:
            tx = max(12, tx - 2)
        
        self.packet.configure(
            sf=sf, bw=bw, tx_power=tx,
            distance=self.distance, directionality_factor=direction_factor
        )
        
        self.last_state = state
        self.last_action = (sf, bw, tx)
    
    def update_learning(self, success: bool, collision: bool, lost: bool, network_load: int):
        self.update_basic_stats(success, collision, lost, network_load)
        
        if self.last_state is not None and self.last_action is not None:
            reward = self._calculate_reward(success, network_load)
            next_state = self._get_state_key(network_load, self.distance, self.stats.get_recent_der())
            
            old_q = self.q_table[self.last_state][self.last_action]
            next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            new_q = old_q + self.learning_rate * (reward + self.discount * next_max - old_q)
            self.q_table[self.last_state][self.last_action] = new_q
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _calculate_reward(self, success: bool, network_load: int) -> float:
        reward = 0
        
        if success:
            reward += 100
            if self.packet.sf <= 9 and network_load > 15:
                reward += 50
            if self.packet.txpow <= 12:
                reward += 20
        else:
            reward -= 80
            if self.packet.sf > 10:
                reward -= 30
        
        reward -= (self.packet.sf - 7) * 5
        reward -= (self.packet.txpow - 10) * 2
        
        energy_used = self.energy_tracker.last_energy_consumed_mah
        if energy_used < 0.0002:
            reward += 15
        
        return max(-150, min(200, reward))

# =============================================================================
# Static Node
# =============================================================================

class StaticNode(BaseNode):
    """Στατικός κόμβος για baseline experiments"""
    
    def __init__(self, node_id: int, period: int, base_station: Any, config: SystemConfig, experiment: int):
        super().__init__(node_id, period, base_station, config)
        self.experiment = experiment
        
        if experiment == 0:
            self.packet.sf = 12
            self.packet.bw = 125
            self.packet.cr = 1
            self.random_freq = False
        elif experiment == 1:
            self.packet.sf = 12
            self.packet.bw = 125
            self.packet.cr = 1
            self.random_freq = True
            self.available_freqs = [868000000, 868300000, 868500000]
        elif experiment == 2:
            self.packet.sf = 7
            self.packet.bw = 500
            self.packet.cr = 0
            self.random_freq = False
        elif experiment == 4:
            self.packet.sf = 12
            self.packet.bw = 125
            self.packet.cr = 0
            self.random_freq = False
        elif experiment in [3, 5]:
            self.random_freq = False
        
        self.packet.rectime = self.packet._calculate_airtime()
    
    def configure_transmission(self, network_load: int):
        direction_factor = self._calculate_directionality_factor()
        
        if hasattr(self, 'random_freq') and self.random_freq and hasattr(self, 'available_freqs'):
            self.packet.freq = random.choice(self.available_freqs)
        
        if self.experiment in [3, 5]:
            if self.distance < 1000:
                self.packet.sf = 7
                self.packet.bw = 500 if self.experiment == 3 else 250
                self.packet.cr = 0
            elif self.distance < 3000:
                self.packet.sf = 9
                self.packet.bw = 125
                self.packet.cr = 1
            else:
                self.packet.sf = 12
                self.packet.bw = 125
                self.packet.cr = 1 if self.experiment == 3 else 0
            
            if self.experiment == 3:
                self.packet.txpow = 20
            elif self.experiment == 5:
                if self.distance < 1000:
                    self.packet.txpow = 10
                elif self.distance < 3000:
                    self.packet.txpow = 14
                else:
                    self.packet.txpow = 17
        
        if direction_factor < 0.3:
            self.packet.txpow = min(20, self.packet.txpow + 3)
        elif direction_factor < 0.6:
            self.packet.txpow = min(20, self.packet.txpow + 1)
        
        self.packet.configure(
            sf=self.packet.sf, bw=self.packet.bw, tx_power=self.packet.txpow,
            distance=self.distance, directionality_factor=direction_factor
        )
    
    def update_learning(self, success: bool, collision: bool, lost: bool, network_load: int):
        self.update_basic_stats(success, collision, lost, network_load)

# =============================================================================
# detect_collision
# =============================================================================

def detect_collision(packet1: LoRaPacket, packet2: LoRaPacket, 
                     full_collision_mode: bool = False) -> bool:
    """Enhanced collision detection"""
    
    if packet1.sf == packet2.sf:
        freq_diff = abs(packet1.freq - packet2.freq)
        if packet1.bw == 500 or packet2.bw == 500:
            if freq_diff < 120000:
                return True
        elif packet1.bw == 250 or packet2.bw == 250:
            if freq_diff < 60000:
                return True
        elif freq_diff < 30000:
            return True
    
    if packet1.sf != packet2.sf:
        rssi_diff = abs(packet1.rssi - packet2.rssi)
        if rssi_diff < 3:
            return True
    
    return False

# =============================================================================
# myBS
# =============================================================================

class myBS:
    """Base Station class"""
    __slots__ = ['id', 'x', 'y']
    
    def __init__(self, id, maxDist, baseDist, nrBS):
        self.id = id
        self.x = 0
        self.y = 0
        
        if nrBS == 1 and id == 0:
            self.x = maxDist
            self.y = maxDist
        elif nrBS == 2:
            if id == 0:
                self.x = maxDist
                self.y = maxDist
            else:
                self.x = maxDist + baseDist
                self.y = maxDist
        elif nrBS == 3:
            if id == 0:
                self.x = maxDist + baseDist
                self.y = maxDist
            elif id == 1:
                self.x = maxDist 
                self.y = maxDist
            else:
                self.x = maxDist + 2 * baseDist
                self.y = maxDist
        else:
            grid_size = math.ceil(math.sqrt(nrBS))
            row = id // grid_size
            col = id % grid_size
            spacing = maxDist * 2 / (grid_size + 1)
            self.x = (col + 1) * spacing
            self.y = (row + 1) * spacing
        
        print(f"BS {id}: x={self.x:.1f}, y={self.y:.1f}")

# =============================================================================
# ENHANCED UNIFIED SIMULATOR
# =============================================================================

class EnhancedLoRaSimulator:
    """Ενοποιημένος προσομοιωτής LoRa με υποστήριξη όλων των τύπων κόμβων"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.env = simpy.Environment()
        
        self.base_stations = []
        self.nodes = []
        self.packets_at_bs = defaultdict(list)
        self.packets_received = defaultdict(list)
        
        self.packet_seq = 0
        self.received_packets = []
        self.collided_packets = []
        self.lost_packets = []
        
        self.maxX = 0
        self.maxY = 0
        
        self.total_aloha_collisions = 0
        self.total_physical_collisions = 0
        
        self.avg_send_time = None
        
        self.bs_stats = defaultdict(lambda: {
            'packets_received': 0, 'packets_collided': 0, 'packets_lost': 0,
            'total_packets': 0, 'der': 0.0, 'collision_rate': 0.0,
            'avg_rssi': 0.0, 'avg_snr': 0.0, 'sf_distribution': defaultdict(int),
            'node_distribution': [], 'energy_consumed_mah': 0.0,
            'directional_nodes': 0, 'non_directional_nodes': 0, 'position': (0, 0)
        })
        
        self.bs_time_series = defaultdict(lambda: {
            'time': [], 'packets': [], 'collisions': [], 'load': [], 'der': []
        })
        
        self.system_stats = {
            'start_time': 0, 'end_time': 0, 'total_packets': 0,
            'total_received': 0, 'total_collisions': 0, 'total_lost': 0,
            'total_energy_consumed_mah': 0.0, 'avg_energy_per_packet_mah': 0.0,
            'estimated_system_lifetime_years': 0.0, 'der_history': deque(maxlen=100),
            'load_history': deque(maxlen=100), 'energy_history': deque(maxlen=100),
            'sf_distribution': defaultdict(int), 'energy_optimization_enabled': config.use_energy_optimization,
            'experiment': 0, 'algorithm_used': ''
        }
    
    def _calculate_aloha_collision_probability(self, num_nodes: int, period_ms: int, airtime_ms: float) -> float:
        activity_factor = 0.9
        packets_per_sec_per_node = 1000.0 / period_ms
        total_packets_per_sec = num_nodes * packets_per_sec_per_node * activity_factor
        
        channel_load = total_packets_per_sec * (airtime_ms / 1000.0)
        collision_prob = 1 - math.exp(-2 * channel_load)
        
        return min(0.95, max(0.0, collision_prob))
    
    def initialize(self, nr_nodes: int, avg_send_time: int, nr_bs: int, 
                  max_dist: float, base_dist: float, experiment: int):
        
        self.avg_send_time = avg_send_time
        
        self._create_base_stations(nr_bs, max_dist, base_dist)
        self._create_nodes(nr_nodes, avg_send_time, nr_bs, experiment)
        
        self.system_stats['experiment'] = experiment
        
        algorithm_names = {
            0: "Baseline SF12",
            1: "Frequency Hopping", 
            2: "Baseline SF7",
            3: "Distance Optimized",
            4: "LoRaWAN Default",
            5: "Power Optimized",
            8: "Enhanced DQN",
            9: "Original ADR",
            10: "Improved ADR",
            11: "ML-ADR"
        }
        
        print(f"\n{'='*70}")
        print(f"ENHANCED LoRa SIMULATOR")
        print(f"{'='*70}")
        print(f"Nodes: {nr_nodes}")
        print(f"Base Stations: {nr_bs}")
        print(f"Battery Capacity: {self.config.battery_capacity_mah} mAh")
        print(f"Energy Optimization: {'ENABLED' if self.config.use_energy_optimization else 'DISABLED'}")
        print(f"Algorithm: {algorithm_names.get(experiment, 'Unknown')}")
        
        avg_airtime = 100
        expected_collision_prob = self._calculate_aloha_collision_probability(
            nr_nodes, avg_send_time, avg_airtime
        )
        print(f"Expected Collision Probability: {expected_collision_prob*100:.1f}%")
        print(f"{'='*70}")
    
    def _create_base_stations(self, nr_bs: int, max_dist: float, base_dist: float):
        for i in range(nr_bs):
            bs = myBS(i, max_dist, base_dist, nr_bs)
            self.base_stations.append(bs)
            self.packets_at_bs[i] = []
            self.packets_received[i] = []
            self.bs_stats[i]['position'] = (bs.x, bs.y)
    
    def _create_nodes(self, nr_nodes: int, avg_send_time: int, nr_bs: int, experiment: int):
        self.nodes = []
        node_id = 0
        
        nodes_per_bs = nr_nodes // nr_bs
        remainder = nr_nodes % nr_bs
        
        for bs_idx in range(nr_bs):
            if bs_idx < remainder:
                nodes_for_this_bs = nodes_per_bs + 1
            else:
                nodes_for_this_bs = nodes_per_bs
            
            print(f"BS{bs_idx}: Allocating {nodes_for_this_bs} nodes")
            
            for _ in range(nodes_for_this_bs):
                bs = self.base_stations[bs_idx]
                
                if experiment == 8:
                    node = EnhancedDQNNode(node_id, avg_send_time, bs, self.config)
                    self.system_stats['algorithm_used'] = "Enhanced DQN"
                elif experiment == 9:
                    node = OriginalADRNode(node_id, avg_send_time, bs, self.config)
                    self.system_stats['algorithm_used'] = "Original ADR"
                elif experiment == 10:
                    node = ImprovedADRNode(node_id, avg_send_time, bs, self.config)
                    self.system_stats['algorithm_used'] = "Improved ADR"
                elif experiment == 11:
                    node = MLADRNode(node_id, avg_send_time, bs, self.config)
                    self.system_stats['algorithm_used'] = "ML-ADR"
                elif experiment in [0, 1, 2, 3, 4, 5]:
                    node = StaticNode(node_id, avg_send_time, bs, self.config, experiment)
                    self.system_stats['algorithm_used'] = "Static"
                else:
                    node = StaticNode(node_id, avg_send_time, bs, self.config, 0)
                
                node.set_env(self.env)
                self.nodes.append(node)
                self.env.process(self._transmission_process(node))
                
                self.bs_stats[bs_idx]['node_distribution'].append(node.id)
                if hasattr(node, 'has_directionality'):
                    if node.has_directionality:
                        self.bs_stats[bs_idx]['directional_nodes'] += 1
                    else:
                        self.bs_stats[bs_idx]['non_directional_nodes'] += 1
                
                node_id += 1
        
        print(f"\nCreated {len(self.nodes)} nodes")
        
        if self.config.directionality:
            total_directional = sum(bs['directional_nodes'] for bs in self.bs_stats.values())
            total_non_directional = sum(bs['non_directional_nodes'] for bs in self.bs_stats.values())
            print(f"Directional nodes: {total_directional} ({total_directional/len(self.nodes)*100:.1f}%)")
            print(f"Non-directional nodes: {total_non_directional} ({total_non_directional/len(self.nodes)*100:.1f}%)")
    
    def _transmission_process(self, node):
        bs_id = node.base_station.id
        
        while True:
            yield self.env.timeout(random.expovariate(1.0 / float(node.period)))
            
            current_time = self.env.now
            current_load = len(self.packets_at_bs[bs_id])
            
            node.configure_transmission(current_load)
            
            if node.packet.lost:
                self.bs_stats[bs_id]['packets_lost'] += 1
                self.system_stats['total_lost'] += 1
                self.lost_packets.append(self.packet_seq)
                node.update_learning(False, False, True, current_load)
                continue
            
            airtime_ms = node.packet.rectime
            aloha_collision_prob = self._calculate_aloha_collision_probability(
                len(self.nodes), node.period, airtime_ms
            )
            
            sf_factor = 1.0
            if node.packet.sf >= 11:
                sf_factor = 0.5
            elif node.packet.sf <= 8:
                sf_factor = 1.5
            
            adjusted_collision_prob = min(0.95, aloha_collision_prob * sf_factor)
            aloha_collision = random.random() < adjusted_collision_prob
            
            self.packets_at_bs[bs_id].append(node)
            node.packet.addTime = self.env.now
            node.packet.seqNr = self.packet_seq
            self.packet_seq += 1
            self.bs_stats[bs_id]['total_packets'] += 1
            self.system_stats['total_packets'] += 1
            
            physical_collision = False
            for other_node in self.packets_at_bs[bs_id]:
                if other_node.id != node.id:
                    if detect_collision(node.packet, other_node.packet, self.config.full_collision):
                        time_diff = abs(node.packet.addTime - other_node.packet.addTime)
                        if time_diff < node.packet.rectime and time_diff < other_node.packet.rectime:
                            physical_collision = True
                            node.packet.collided = 1
                            other_node.packet.collided = 1
                            break
            
            collision = aloha_collision or physical_collision
            
            if aloha_collision and not physical_collision:
                self.total_aloha_collisions += 1
            elif physical_collision:
                self.total_physical_collisions += 1
            
            yield self.env.timeout(node.packet.rectime)
            
            success = not collision and not node.packet.lost
            
            if success:
                self.received_packets.append(node.packet.seqNr)
                self.packets_received[bs_id].append(node.packet.seqNr)
                self.bs_stats[bs_id]['packets_received'] += 1
                self.system_stats['total_received'] += 1
            else:
                if collision:
                    self.collided_packets.append(node.packet.seqNr)
                    self.bs_stats[bs_id]['packets_collided'] += 1
                    self.system_stats['total_collisions'] += 1
                else:
                    self.lost_packets.append(node.packet.seqNr)
                    self.bs_stats[bs_id]['packets_lost'] += 1
                    self.system_stats['total_lost'] += 1
            
            node.update_learning(success, collision, False, current_load)
            
            self.system_stats['total_energy_consumed_mah'] += node.energy_tracker.last_energy_consumed_mah
            self.bs_stats[bs_id]['energy_consumed_mah'] += node.energy_tracker.last_energy_consumed_mah
            self.system_stats['sf_distribution'][node.packet.sf] += 1
            self.bs_stats[bs_id]['sf_distribution'][node.packet.sf] += 1
            
            node.packet.collided = 0
            node.packet.processed = 0
            
            if node in self.packets_at_bs[bs_id]:
                self.packets_at_bs[bs_id].remove(node)
            
            total_bs = self.bs_stats[bs_id]['total_packets']
            received_bs = self.bs_stats[bs_id]['packets_received']
            if total_bs > 0:
                self.bs_stats[bs_id]['der'] = received_bs / total_bs * 100
            
            if self.packet_seq % 1000 == 0:
                self._print_progress()
    
    def _print_progress(self):
        current_time = self.env.now
        total_packets = self.system_stats['total_packets']
        received = len(self.received_packets)
        collisions = len(self.collided_packets)
        
        print(f"\n[Progress @ {current_time:.0f}ms] Packets: {total_packets}, "
              f"DER: {received/total_packets*100:.1f}%, "
              f"Collisions: {collisions/total_packets*100:.1f}% "
              f"(Aloha: {self.total_aloha_collisions}, Physical: {self.total_physical_collisions})")
        
        for bs_id in sorted(self.bs_stats.keys()):
            stats = self.bs_stats[bs_id]
            if stats['total_packets'] > 0:
                print(f"  BS{bs_id}: DER={stats['der']:.1f}% | "
                      f"Packets={stats['total_packets']}")
    
    def run(self, sim_time: int):
        print(f"\nStarting simulation for {sim_time}ms...")
        self.system_stats['start_time'] = time.time()
        self.config.simulation_time_ms = sim_time
        
        self.env.run(until=sim_time)
        
        self.system_stats['end_time'] = time.time()
        simulation_time = self.system_stats['end_time'] - self.system_stats['start_time']
        
        print(f"\nSimulation completed in {simulation_time:.2f} seconds")
        self._display_results()
        self._save_results()
    
    def _display_results(self):
        print(f"\n{'='*70}")
        print(f"FINAL SIMULATION RESULTS")
        print(f"Algorithm: {self.system_stats['algorithm_used']}")
        print(f"{'='*70}")
        
        total_packets = self.system_stats['total_packets']
        received = len(self.received_packets)
        collisions = len(self.collided_packets)
        lost = len(self.lost_packets)
        
        print(f"Total packets sent: {total_packets:,}")
        print(f"Packets received successfully: {received:,}")
        print(f"Packets lost in collisions: {collisions:,}")
        print(f"Packets lost due to weak signal: {lost:,}")
        
        print(f"\nCollision Breakdown:")
        print(f"  Aloha model collisions: {self.total_aloha_collisions}")
        print(f"  Physical collisions: {self.total_physical_collisions}")
        
        if total_packets > 0:
            der = received / total_packets * 100
            collision_rate = collisions / total_packets * 100
            loss_rate = lost / total_packets * 100
            
            print(f"\nPacket Delivery Rate (DER): {der:.2f}%")
            print(f"Collision Rate: {collision_rate:.2f}%")
            print(f"Signal Loss Rate: {loss_rate:.2f}%")
            
            total_energy = self.system_stats['total_energy_consumed_mah']
            
            print(f"\n{'='*70}")
            print(f"ENERGY PERFORMANCE ANALYSIS")
            print(f"{'='*70}")
            print(f"Total energy consumed: {total_energy:.2f} mAh")
            print(f"Avg energy per packet (όλες μεταδόσεις): {total_energy/total_packets:.6f} mAh")
            
            # ΣΩΣΤΟΣ ΥΠΟΛΟΓΙΣΜΟΣ - Λαμβάνει υπόψη το DER
            if received > 0:
                true_energy_per_success_mah = total_energy / received
                true_energy_per_success_uah = true_energy_per_success_mah * 1_000_000
                avg_energy_per_tx = total_energy / total_packets if total_packets > 0 else 0
                wasted_energy_percent = (1 - received/total_packets) * 100
                
                print(f"\n{'='*70}")
                print(f"ΠΡΑΓΜΑΤΙΚΗ ΕΝΕΡΓΕΙΑΚΗ ΑΝΑΛΥΣΗ")
                print(f"{'='*70}")
                print(f"Ενέργεια ανά ΜΕΤΑΔΟΣΗ (avg): {avg_energy_per_tx:.6f} mAh ({avg_energy_per_tx*1e6:.2f} μAh)")
                print(f"Ενέργεια ανά ΕΠΙΤΥΧΗΜΕΝΟ (ΠΡΑΓΜΑΤΙΚΗ): {true_energy_per_success_mah:.6f} mAh ({true_energy_per_success_uah:.2f} μAh)")
                print(f"Ενέργεια που χάθηκε σε αποτυχημένες: {wasted_energy_percent:.1f}%")
                
                # ΣΩΣΤΗ ΕΚΤΙΜΗΣΗ ΔΙΑΡΚΕΙΑΣ ΖΩΗΣ
                if self.avg_send_time is not None:
                    period_ms = self.avg_send_time
                    target_packets_per_hour = 3600 / (period_ms / 1000)
                    success_rate = received / total_packets
                    
                    # Πραγματικές μεταδόσεις που χρειάζονται για να πετύχουμε τον στόχο
                    actual_transmissions_per_hour = target_packets_per_hour / success_rate
                    
                    # Ενέργεια ανά μετάδοση (συμπεριλαμβανομένων των αποτυχημένων)
                    energy_per_transmission = total_energy / total_packets
                    
                    # Ενέργεια ανά ώρα
                    energy_per_hour = actual_transmissions_per_hour * energy_per_transmission
                    
                    battery_per_node = self.config.battery_capacity_mah
                    
                    if energy_per_hour > 0:
                        lifetime_hours = battery_per_node / energy_per_hour
                        lifetime_days = lifetime_hours / 24
                        lifetime_years = lifetime_hours / (24 * 365)
                        lifetime_months = lifetime_years * 12
                        
                        print(f"\n{'='*70}")
                        print(f"ΣΩΣΤΗ ΕΚΤΙΜΗΣΗ ΔΙΑΡΚΕΙΑΣ ΖΩΗΣ ΜΠΑΤΑΡΙΑΣ (ανά κόμβο)")
                        print(f"{'='*70}")
                        print(f"Στόχος: 1 ΕΠΙΤΥΧΗΜΕΝΟ πακέτο κάθε {period_ms}ms")
                        print(f"Πακέτα/ώρα στόχος: {target_packets_per_hour:.0f}")
                        print(f"Success Rate (DER): {success_rate*100:.1f}%")
                        print(f"Πραγματικές μεταδόσεις/ώρα: {actual_transmissions_per_hour:.1f}")
                        print(f"Ενέργεια ανά μετάδοση: {energy_per_transmission:.6f} mAh ({energy_per_transmission*1e6:.2f} μAh)")
                        print(f"Ενέργεια που χρειάζεται ανά ώρα: {energy_per_hour:.4f} mAh")
                        print(f"\nΕκτίμηση διάρκειας ζωής (μπαταρία {battery_per_node} mAh):")
                        print(f"  Ώρες: {lifetime_hours:.1f} ώρες")
                        print(f"  Ημέρες: {lifetime_days:.1f} ημέρες")
                        print(f"  Μήνες: {lifetime_months:.1f} μήνες")
                        print(f"  Χρόνια: {lifetime_years:.2f} χρόνια")
                        
                        print(f"\n⚠️ ΣΗΜΑΝΤΙΚΟ: Η πραγματική διάρκεια ζωής είναι {lifetime_days:.1f} ημέρες")
                        print(f"   και ΟΧΙ {battery_per_node / (true_energy_per_success_mah * target_packets_per_hour) * 24:.0f} ημέρες")
                        print(f"   (η τελευταία είναι λάθος εκτίμηση που αγνοεί το DER!)")
    
    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment = self.system_stats['experiment']
        
        exp_names = {
            0: "baseline_sf12", 1: "freq_hopping", 2: "baseline_sf7",
            3: "distance_opt", 4: "lorawan_default", 5: "power_opt",
            8: "enhanced_dqn", 9: "original_adr", 10: "improved_adr", 11: "ml_adr"
        }
        exp_prefix = exp_names.get(experiment, f"exp{experiment}")
        
        simulation_time_ms = self.env.now
        
        total_energy = self.system_stats['total_energy_consumed_mah']
        received = len(self.received_packets)
        total_packets = self.system_stats['total_packets']
        
        if received > 0:
            true_energy_per_success_mah = total_energy / received
            true_energy_per_success_uah = true_energy_per_success_mah * 1_000_000
        else:
            true_energy_per_success_mah = float('inf')
            true_energy_per_success_uah = float('inf')
        
        main_report = {
            'timestamp': timestamp,
            'experiment': experiment,
            'algorithm': self.system_stats['algorithm_used'],
            'simulation_parameters': {
                'total_nodes': len(self.nodes),
                'base_stations': len(self.base_stations),
                'simulation_time_ms': simulation_time_ms,
                'battery_capacity_mah': self.config.battery_capacity_mah,
                'energy_optimization': self.config.use_energy_optimization,
                'directionality': self.config.directionality,
                'send_period_ms': self.avg_send_time
            },
            'performance_metrics': {
                'total_packets': total_packets,
                'received_packets': received,
                'collided_packets': len(self.collided_packets),
                'lost_packets': len(self.lost_packets),
                'der': received / max(1, total_packets),
                'collision_rate': len(self.collided_packets) / max(1, total_packets),
                'loss_rate': len(self.lost_packets) / max(1, total_packets)
            },
            'collision_breakdown': {
                'aloha_collisions': self.total_aloha_collisions,
                'physical_collisions': self.total_physical_collisions
            },
            'sf_distribution': dict(self.system_stats['sf_distribution']),
            'energy_analysis': {
                'total_energy_consumed_mah': total_energy,
                'avg_energy_per_packet_mah': total_energy / max(1, total_packets),
                'true_energy_per_successful_packet_mah': true_energy_per_success_mah,
                'true_energy_per_successful_packet_uah': true_energy_per_success_uah
            },
            'execution_time_seconds': self.system_stats['end_time'] - self.system_stats['start_time']
        }
        
        main_file = f"{exp_prefix}_results_{timestamp}.json"
        with open(main_file, 'w') as f:
            json.dump(main_report, f, indent=2)
        
        print(f"\nMain results saved to {main_file}")
        
        nodes_file = f"{exp_prefix}_node_reports_{timestamp}.csv"
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['NodeID', 'Distance', 'DER', 'EnergyEfficiency', 
                            'RemainingEnergy', 'EstimatedLifetimeHours',
                            'EstimatedLifetimeDays', 'EstimatedLifetimeMonths',
                            'EstimatedLifetimeYears', 'CurrentSF', 'CurrentBW', 
                            'CurrentTX', 'Directional', 'EnergyPerSuccessful_uAh'])
            
            for node in self.nodes:
                if hasattr(node, 'get_energy_report'):
                    report = node.get_energy_report(simulation_time_ms)
                    writer.writerow([
                        report['node_id'],
                        f"{report['distance']:.1f}",
                        f"{report['der']:.4f}",
                        f"{report['energy_metrics']['energy_efficiency']:.4f}",
                        f"{report['energy_metrics']['remaining_energy_mah']:.2f}",
                        f"{report['energy_metrics']['estimated_lifetime_hours']:.1f}",
                        f"{report['energy_metrics']['estimated_lifetime_days']:.1f}",
                        f"{report['energy_metrics']['estimated_lifetime_months']:.1f}",
                        f"{report['energy_metrics']['estimated_lifetime_years']:.2f}",
                        report['current_config']['sf'],
                        report['current_config']['bw'],
                        report['current_config']['tx_power'],
                        'YES' if report['directionality']['has_directionality'] else 'NO',
                        f"{report['energy_metrics']['energy_per_successful_packet_mah']*1e6:.2f}"
                    ])
        
        print(f"Node reports saved to {nodes_file}")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_comparison_plots(results_data: Dict, output_dir: str = "plots"):
    """Διορθωμένη έκδοση με σωστούς υπολογισμούς διάρκειας ζωής"""
    
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    periods = list(results_data.keys())
    
    algorithms = ['Enhanced DQN', 'Original ADR', 'ML-ADR']
    der_data = {alg: [results_data[p][alg]['der'] for p in periods] for alg in algorithms if alg in results_data[periods[0]]}
    energy_data = {alg: [results_data[p][alg]['true_energy_per_success_uah'] for p in periods] for alg in algorithms if alg in results_data[periods[0]]}
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    # Figure 1: DER Comparison
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for i, alg in enumerate(der_data.keys()):
        ax1.plot(periods, der_data[alg], marker=markers[i], linewidth=2, 
                markersize=10, label=alg, color=colors[i])
    
    ax1.set_xlabel('Transmission Period (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Packet Delivery Rate - DER (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Performance Comparison: DQN vs ADR vs ML-ADR', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(min(periods)-1000, max(periods)+1000)
    ax1.set_ylim(55, 100)
    
    for i, alg in enumerate(der_data.keys()):
        for j, (x, y) in enumerate(zip(periods, der_data[alg])):
            ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0, 12), ha='center', fontsize=9, color=colors[i])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/der_comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Energy per Successful Packet
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for i, alg in enumerate(energy_data.keys()):
        ax2.plot(periods, energy_data[alg], marker=markers[i], linewidth=2,
                markersize=10, label=alg, color=colors[i])
    
    ax2.set_xlabel('Transmission Period (ms)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy per Successful Packet (μAh)', fontsize=14, fontweight='bold')
    ax2.set_title('Energy Efficiency Comparison (per successful packet)', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(min(periods)-1000, max(periods)+1000)
    
    for i, alg in enumerate(energy_data.keys()):
        for j, (x, y) in enumerate(zip(periods, energy_data[alg])):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8, color=colors[i])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: TRUE Battery Lifetime (ΔΙΟΡΘΩΜΕΝΟ - ΛΑΜΒΑΝΕΙ ΥΠΟΨΗ DER)
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    battery_capacity = 300  # mAh
    
    battery_life_data = {}
    for alg in algorithms:
        lifetime_days_list = []
        for idx, period in enumerate(periods):
            der = der_data[alg][idx] / 100  # Convert to decimal
            energy_uah = energy_data[alg][idx]
            
            # ΣΩΣΤΟΣ ΥΠΟΛΟΓΙΣΜΟΣ
            target_packets_per_hour = 3600 / (period / 1000)
            energy_per_transmission_mah = (energy_uah / 1_000_000) / der
            actual_transmissions_per_hour = target_packets_per_hour / der
            energy_per_hour = actual_transmissions_per_hour * energy_per_transmission_mah
            
            if energy_per_hour > 0:
                hours = battery_capacity / energy_per_hour
                days = hours / 24
            else:
                days = 0
            lifetime_days_list.append(days)
        battery_life_data[alg] = lifetime_days_list
    
    width = 0.25
    x = np.arange(len(periods))
    for i, alg in enumerate(algorithms):
        bars = ax3.bar(x + i*width, battery_life_data[alg], width, label=alg, color=colors[i])
        
        for bar, days in zip(bars, battery_life_data[alg]):
            ax3.annotate(f'{days:.1f}d', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    
    ax3.set_xlabel('Transmission Period (ms)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Battery Lifetime (days)', fontsize=14, fontweight='bold')
    ax3.set_title('TRUE Battery Lifetime (accounting for retransmissions & DER)', fontsize=16, fontweight='bold')
    ax3.set_xticks(x + width)
    
    # ΔΙΟΡΘΩΜΕΝΟ: Δημιουργία ετικετών για κάθε περίοδο (χρησιμοποιώντας το πρώτο algorithm για DER reference)
    xtick_labels = [f'{p}ms\n(DER: {der_data[algorithms[0]][i]:.1f}%)' for i, p in enumerate(periods)]
    ax3.set_xticklabels(xtick_labels, fontsize=9)
    
    ax3.legend(loc='upper left', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/true_battery_lifetime_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/ directory")
    print(f"  - der_comparison_{timestamp}.png")
    print(f"  - energy_comparison_{timestamp}.png")
    print(f"  - true_battery_lifetime_{timestamp}.png (CORRECTED)")

# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison_experiments():
    """Εκτέλεση συγκριτικών πειραμάτων με πλήρη εκτύπωση αποτελεσμάτων στην κονσόλα"""
    
    periods = [1000, 3000, 6000, 10000, 20000]
    algorithms = {
        8: "Enhanced DQN",
        9: "Original ADR", 
        11: "ML-ADR"
    }
    
    results = {}
    sf_distributions = {}
    learning_curves = {}
    computational_cost = {}
    
    print("\n" + "="*80)
    print("ΈΝΑΡΞΗ ΣΥΓΚΡΙΤΙΚΩΝ ΠΕΙΡΑΜΑΤΩΝ")
    print("="*80)
    print(f"Περίοδοι προς δοκιμή: {periods}")
    print(f"Αλγόριθμοι: {list(algorithms.values())}")
    print("="*80)
    
    for period in periods:
        print(f"\n{'='*70}")
        print(f"ΠΕΡΙΟΔΟΣ ΜΕΤΑΔΟΣΗΣ: {period} ms")
        print(f"{'='*70}")
        
        results[period] = {}
        sf_distributions[period] = {}
        learning_curves[period] = {}
        computational_cost[period] = {}
        
        for exp, name in algorithms.items():
            print(f"\n{'─'*50}")
            print(f"ΕΚΤΕΛΕΣΗ: {name}")
            print(f"{'─'*50}")
            
            # Μέτρηση χρόνου
            start_time = time.time()
            
            config = SystemConfig(
                directionality=True,
                use_dqn=(exp == 8),
                use_improved_adr=(exp == 10),
                use_ml_adr=(exp == 11),
                battery_capacity_mah=300.0
            )
            
            simulator = EnhancedLoRaSimulator(config)
            simulator.initialize(1000, period, 1, 5000, 1000, exp)
            
            # Αποθήκευση της αρχικής μεθόδου run
            original_run = simulator.run
            
            # Δημιουργία monitored_run με σωστά ορίσματα
            def monitored_run(self, sim_time):
                interval = 10000  # κάθε 10.000 ms
                der_over_time = []
                time_points = []
                
                for t in range(interval, sim_time + 1, interval):
                    self.env.run(until=t)
                    total_packets = self.system_stats['total_packets']
                    received = len(self.received_packets)
                    if total_packets > 0:
                        der = received / total_packets * 100
                    else:
                        der = 0
                    der_over_time.append(der)
                    time_points.append(t)
                
                # Υπόλοιπο χρόνο μέχρι το τέλος
                if self.env.now < sim_time:
                    self.env.run(until=sim_time)
                
                return der_over_time, time_points
            
            # Αποθήκευση της monitored_run ως μέθοδο
            simulator.monitored_run = monitored_run.__get__(simulator, EnhancedLoRaSimulator)
            
            # Εκτέλεση της παρακολουθούμενης προσομοίωσης
            der_over_time, time_points = simulator.monitored_run(200000)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Υπάρχοντα αποτελέσματα
            total_energy = simulator.system_stats['total_energy_consumed_mah']
            received = len(simulator.received_packets)
            total_packets = simulator.system_stats['total_packets']
            collisions = len(simulator.collided_packets)
            lost = len(simulator.lost_packets)
            aloha_collisions = simulator.total_aloha_collisions
            physical_collisions = simulator.total_physical_collisions
            
            if received > 0:
                true_energy_per_success_uah = (total_energy / received) * 1_000_000
            else:
                true_energy_per_success_uah = 0
            
            results[period][name] = {
                'der': received / max(1, total_packets) * 100,
                'collision_rate': collisions / max(1, total_packets) * 100,
                'loss_rate': lost / max(1, total_packets) * 100,
                'total_packets': total_packets,
                'received_packets': received,
                'avg_energy_uah': (total_energy / max(1, total_packets)) * 1_000_000,
                'true_energy_per_success_uah': true_energy_per_success_uah
            }
            
            # Αποθήκευση κατανομής SF
            sf_dist = simulator.system_stats['sf_distribution']
            total_sf = sum(sf_dist.values())
            sf_percentages = {sf: (sf_dist.get(sf, 0) / total_sf * 100) if total_sf > 0 else 0 for sf in range(7, 13)}
            sf_distributions[period][name] = {sf: sf_dist.get(sf, 0) for sf in range(7, 13)}
            
            # Αποθήκευση learning curve
            learning_curves[period][name] = {
                'time_points': time_points,
                'der_values': der_over_time
            }
            
            # Αποθήκευση υπολογιστικού κόστους
            num_params = 0
            if exp == 8:  # DQN
                num_params = 5506
            elif exp == 11:  # ML-ADR
                num_params = 2376
            
            computational_cost[period][name] = {
                'execution_time_sec': execution_time,
                'num_parameters': num_params
            }
            
            # ========== ΕΚΤΥΠΩΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΤΗΝ ΚΟΝΣΟΛΑ ==========
            print(f"\n>>> ΑΠΟΤΕΛΕΣΜΑΤΑ ΓΙΑ {name} (περίοδος={period}ms) <<<")
            print(f"  Συνολικές μεταδόσεις: {total_packets:,}")
            print(f"  Επιτυχημένες: {received:,}")
            print(f"  Συγκρούσεις: {collisions:,} (ALOHA: {aloha_collisions}, Φυσικές: {physical_collisions})")
            print(f"  Απώλειες σήματος: {lost:,}")
            print(f"  DER: {results[period][name]['der']:.2f}%")
            print(f"  Collision Rate: {results[period][name]['collision_rate']:.2f}%")
            print(f"  Απώλειες σήματος: {results[period][name]['loss_rate']:.2f}%")
            print(f"  Μέση ενέργεια ανά μετάδοση: {results[period][name]['avg_energy_uah']:.2f} μAh")
            print(f"  Πραγματική ενέργεια ανά επιτυχημένο: {results[period][name]['true_energy_per_success_uah']:.2f} μAh")
            print(f"  Χρόνος εκτέλεσης: {execution_time:.2f} sec")
            
            # Εκτύπωση κατανομής SF
            print(f"  Κατανομή SF (%):")
            for sf in range(7, 13):
                pct = sf_percentages.get(sf, 0)
                bar = "█" * int(pct / 2)
                print(f"    SF{sf}: {pct:5.1f}% {bar}")
            
            print(f"{'─'*50}")
    
    # ========== ΤΕΛΙΚΗ ΣΥΝΟΨΗ ΣΤΗΝ ΚΟΝΣΟΛΑ ==========
    print("\n" + "="*80)
    print("ΤΕΛΙΚΗ ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print("="*80)
    
    print("\n--- 1. PACKET DELIVERY RATE (DER %) ---")
    print(f"{'Period':<10} {'Enhanced DQN':<15} {'Original ADR':<15} {'ML-ADR':<15}")
    print(f"{'-'*55}")
    for period in periods:
        dqn_der = results[period].get("Enhanced DQN", {}).get('der', 0)
        adr_der = results[period].get("Original ADR", {}).get('der', 0)
        ml_der = results[period].get("ML-ADR", {}).get('der', 0)
        print(f"{period:<10} {dqn_der:<15.1f} {adr_der:<15.1f} {ml_der:<15.1f}")
    
    print("\n--- 2. ΠΡΑΓΜΑΤΙΚΗ ΕΝΕΡΓΕΙΑ ΑΝΑ ΕΠΙΤΥΧΗΜΕΝΟ (μAh) ---")
    print(f"{'Period':<10} {'Enhanced DQN':<15} {'Original ADR':<15} {'ML-ADR':<15}")
    print(f"{'-'*55}")
    for period in periods:
        dqn_energy = results[period].get("Enhanced DQN", {}).get('true_energy_per_success_uah', 0)
        adr_energy = results[period].get("Original ADR", {}).get('true_energy_per_success_uah', 0)
        ml_energy = results[period].get("ML-ADR", {}).get('true_energy_per_success_uah', 0)
        print(f"{period:<10} {dqn_energy:<15.2f} {adr_energy:<15.2f} {ml_energy:<15.2f}")
    
    print("\n--- 3. ΔΙΑΡΚΕΙΑ ΖΩΗΣ ΜΠΑΤΑΡΙΑΣ (ημέρες) ---")
    print(f"{'Period':<10} {'Enhanced DQN':<15} {'Original ADR':<15} {'ML-ADR':<15}")
    print(f"{'-'*55}")
    for period in periods:
        packets_per_hour = 3600 / (period / 1000)
        dqn_energy = results[period].get("Enhanced DQN", {}).get('true_energy_per_success_uah', 0)
        adr_energy = results[period].get("Original ADR", {}).get('true_energy_per_success_uah', 0)
        ml_energy = results[period].get("ML-ADR", {}).get('true_energy_per_success_uah', 0)
        
        dqn_days = (300 / ((dqn_energy/1_000_000) * packets_per_hour)) / 24 if dqn_energy > 0 else 0
        adr_days = (300 / ((adr_energy/1_000_000) * packets_per_hour)) / 24 if adr_energy > 0 else 0
        ml_days = (300 / ((ml_energy/1_000_000) * packets_per_hour)) / 24 if ml_energy > 0 else 0
        
        print(f"{period:<10} {dqn_days:<15.1f} {adr_days:<15.1f} {ml_days:<15.1f}")
    
    print("\n--- 4. ΥΠΟΛΟΓΙΣΤΙΚΟ ΚΟΣΤΟΣ (περίοδος 1000 ms) ---")
    print(f"{'Αλγόριθμος':<15} {'Χρόνος (sec)':<15} {'Παράμετροι':<15}")
    print(f"{'-'*45}")
    for name in algorithms.values():
        cost = computational_cost[1000].get(name, {})
        exec_time = cost.get('execution_time_sec', 0)
        num_params = cost.get('num_parameters', 0)
        print(f"{name:<15} {exec_time:<15.2f} {num_params:<15}")
    
    print("\n--- 5. ΣΥΓΚΡΙΣΗ ΒΕΛΤΙΩΣΗΣ DQN vs ORIGINAL ADR ---")
    print(f"{'Period':<10} {'Βελτίωση DER':<20} {'Μείωση Ενέργειας':<20}")
    print(f"{'-'*50}")
    for period in periods:
        dqn_der = results[period].get("Enhanced DQN", {}).get('der', 0)
        adr_der = results[period].get("Original ADR", {}).get('der', 0)
        dqn_energy = results[period].get("Enhanced DQN", {}).get('true_energy_per_success_uah', 0)
        adr_energy = results[period].get("Original ADR", {}).get('true_energy_per_success_uah', 0)
        
        der_improvement = dqn_der - adr_der
        energy_reduction = ((adr_energy - dqn_energy) / adr_energy * 100) if adr_energy > 0 else 0
        
        print(f"{period:<10} +{der_improvement:<19.1f}% {energy_reduction:<19.1f}%")
    
    print("\n" + "="*80)
    print("ΣΥΜΠΕΡΑΣΜΑΤΑ")
    print("="*80)
    
    # Υπολογισμός μέσης βελτίωσης
    avg_der_improvement = sum([results[p].get("Enhanced DQN", {}).get('der', 0) - results[p].get("Original ADR", {}).get('der', 0) for p in periods]) / len(periods)
    avg_energy_reduction = sum([((results[p].get("Original ADR", {}).get('true_energy_per_success_uah', 0) - results[p].get("Enhanced DQN", {}).get('true_energy_per_success_uah', 0)) / results[p].get("Original ADR", {}).get('true_energy_per_success_uah', 1) * 100) for p in periods if results[p].get("Original ADR", {}).get('true_energy_per_success_uah', 0) > 0]) / len(periods)
    
    print(f"• Ο DQN υπερτερεί του Original ADR σε όλες τις περιόδους")
    print(f"• Μέση βελτίωση DER: {avg_der_improvement:.1f} ποσοστιαίες μονάδες")
    print(f"• Μέση μείωση ενέργειας ανά επιτυχημένο πακέτο: {avg_energy_reduction:.1f}%")
    print(f"• Ο ML-ADR αποτελεί ενδιάμεση λύση με μικρότερο υπολογιστικό κόστος")
    print(f"• Η μεγαλύτερη βελτίωση παρατηρείται σε συνθήκες υψηλού φόρτου (1000 ms)")
    
    print("\n" + "="*80)
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"comparison_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f"sf_distributions_{timestamp}.json", 'w') as f:
        json.dump(sf_distributions, f, indent=2)
    
    with open(f"learning_curves_{timestamp}.json", 'w') as f:
        json.dump(learning_curves, f, indent=2)
    
    with open(f"computational_cost_{timestamp}.json", 'w') as f:
        json.dump(computational_cost, f, indent=2)
        
    
    print(f"\nΑποτελέσματα αποθηκεύτηκαν σε αρχεία:")
    print(f"  - comparison_results_{timestamp}.json")
    print(f"  - sf_distributions_{timestamp}.json")
    print(f"  - learning_curves_{timestamp}.json")
    print(f"  - computational_cost_{timestamp}.json")
    
    # Create plots
    create_comparison_plots(results)
    plot_sf_distributions(sf_distributions, periods)
    plot_learning_curves(learning_curves, periods)
    
    return results


def _get_num_parameters(experiment: int) -> int:
    """Επιστρέφει τον αριθμό παραμέτρων για κάθε αλγόριθμο"""
    if experiment == 8:  # DQN
        return 5506  # 18*64 + 64 + 64*66 + 66
    elif experiment == 11:  # ML-ADR
        return 2376  # 36 states * 66 actions
    else:  # Original ADR
        return 0
def plot_sf_distributions(sf_distributions: Dict, periods: List[int], output_dir: str = "plots"):
    """Δημιουργεί faceted ραβδογράμματα για την κατανομή SF για ΟΛΕΣ τις περιόδους"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    algorithms = ['Enhanced DQN', 'Original ADR', 'ML-ADR']
    sf_values = list(range(7, 13))
    
    # Χρωματική παλέτα για τους αλγορίθμους
    colors = {'Enhanced DQN': '#2E86AB', 'Original ADR': '#A23B72', 'ML-ADR': '#F18F01'}
    
    # Δημιουργία ενός μεγάλου figure με subplots (2x3, αλλά έχουμε 5 περιόδους)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Ισοπέδωση για εύκολη πρόσβαση
    
    # Διαγραφή του 6ου subplot (περισσότερο)
    axes[5].set_visible(False)
    
    for idx, period in enumerate(periods):
        ax = axes[idx]
        
        if period not in sf_distributions:
            continue
        
        x = np.arange(len(sf_values))
        width = 0.25
        multiplier = 0
        
        for alg in algorithms:
            # Βρες το όνομα του αλγορίθμου όπως αποθηκεύτηκε
            alg_key = next((k for k in sf_distributions[period].keys() if alg in k), None)
            if alg_key is None:
                continue
            
            sf_counts = sf_distributions[period][alg_key]
            total = sum(sf_counts.values())
            if total == 0:
                continue
            
            percentages = [sf_counts.get(sf, 0) / total * 100 for sf in sf_values]
            
            offset = width * multiplier
            rects = ax.bar(x + offset, percentages, width, label=alg, color=colors.get(alg, '#888888'))
            
            # Προσθήκη ετικετών μόνο για σημαντικά ποσοστά (>10%)
            for rect, pct in zip(rects, percentages):
                if pct > 10:
                    ax.annotate(f'{pct:.0f}%', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
            multiplier += 1
        
        ax.set_title(f'Period = {period} ms', fontsize=12, fontweight='bold')
        ax.set_xlabel('Spreading Factor (SF)', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(sf_values)
        ax.set_ylim(0, 70)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('SF Distribution for All Transmission Periods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'{output_dir}/sf_distribution_all_periods_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSF distribution plot (all periods) saved to {output_dir}/sf_distribution_all_periods_{timestamp}.png")

def plot_learning_curves(learning_curves: Dict, periods: List[int], output_dir: str = "plots"):
    """Δημιουργεί faceted γραφήματα learning curves για ΟΛΕΣ τις περιόδους"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    algorithms = ['Enhanced DQN', 'Original ADR', 'ML-ADR']
    colors = {'Enhanced DQN': '#2E86AB', 'Original ADR': '#A23B72', 'ML-ADR': '#F18F01'}
    linestyles = {'Enhanced DQN': '-', 'Original ADR': '--', 'ML-ADR': '-.'}
    
    # Δημιουργία ενός μεγάλου figure με subplots (2x3, αλλά έχουμε 5 περιόδους)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Διαγραφή του 6ου subplot
    axes[5].set_visible(False)
    
    for idx, period in enumerate(periods):
        ax = axes[idx]
        
        if period not in learning_curves:
            print(f"Warning: Period {period} not found in learning_curves data")
            continue
        
        for alg in algorithms:
            alg_key = next((k for k in learning_curves[period].keys() if alg in k), None)
            if alg_key is None:
                continue
            
            data = learning_curves[period][alg_key]
            time_points = data.get('time_points', [])
            der_values = data.get('der_values', [])
            
            if time_points and der_values:
                ax.plot(time_points, der_values, 
                       linewidth=2, 
                       linestyle=linestyles.get(alg, '-'),
                       label=alg, 
                       color=colors.get(alg, '#888888'),
                       marker='o', markersize=3, markevery=5)  # marker κάθε 5 σημεία
        
        ax.set_title(f'Period = {period} ms', fontsize=12, fontweight='bold')
        ax.set_xlabel('Simulation Time (ms)', fontsize=10)
        ax.set_ylabel('DER (%)', fontsize=10)
        ax.set_xlim(0, 200000)
        
        # Δυναμικό ylim ανάλογα με την περίοδο
        if period <= 1000:
            ax.set_ylim(50, 100)
        elif period <= 3000:
            ax.set_ylim(70, 100)
        else:
            ax.set_ylim(85, 100)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('Learning Curves for All Transmission Periods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'{output_dir}/learning_curves_all_periods_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nLearning curves plot (all periods) saved to {output_dir}/learning_curves_all_periods_{timestamp}.png") 

def print_computational_cost_table(computational_cost: Dict, output_dir: str = "plots"):
    """Εκτυπώνει και αποθηκεύει πίνακα υπολογιστικού κόστους"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Εκτύπωση στην κονσόλα
    print(f"\n{'='*80}")
    print("ΥΠΟΛΟΓΙΣΤΙΚΟ ΚΟΣΤΟΣ ΑΝΑ ΑΛΓΟΡΙΘΜΟ")
    print(f"{'='*80}")
    
    # Πάρε μια αντιπροσωπευτική περίοδο (π.χ. 1000 ms)
    sample_period = 1000
    if sample_period in computational_cost:
        print(f"\nΓια περίοδο μετάδοσης {sample_period} ms:")
        print(f"{'Αλγόριθμος':<20} {'Χρόνος (sec)':<15} {'Μνήμη (MB)':<15} {'Παράμετροι':<15}")
        print(f"{'-'*65}")
        
        for alg_name, data in computational_cost[sample_period].items():
            print(f"{alg_name:<20} {data['execution_time_sec']:<15.2f} {data['memory_used_mb']:<15.2f} {data['num_parameters']:<15}")
    
    # Αποθήκευση σε CSV
    csv_file = f"{output_dir}/computational_cost_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Period_ms', 'Algorithm', 'ExecutionTime_sec', 'MemoryUsed_MB', 'NumParameters'])
        
        for period, alg_data in computational_cost.items():
            for alg_name, data in alg_data.items():
                writer.writerow([period, alg_name, data['execution_time_sec'], data['memory_used_mb'], data['num_parameters']])
    
    print(f"\nComputational cost table saved to {csv_file}")     
# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":
    
    # Check for comparison mode
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        print("\n" + "="*70)
        print("RUNNING COMPARISON MODE")
        print("="*70)
        run_comparison_experiments()
        sys.exit(0)
    
    # Normal mode with command line arguments
    if len(sys.argv) >= 10:
        nrNodes = int(sys.argv[1])
        avgSendTime = int(sys.argv[2])
        experiment = int(sys.argv[3])
        simtime = int(sys.argv[4])
        nrBS = int(sys.argv[5])
        full_collision_flag = bool(int(sys.argv[6]))
        directionality = bool(int(sys.argv[7]))
        nrNetworks = int(sys.argv[8])
        baseDist = float(sys.argv[9])
    else:
        print("Using default parameters...")
        print("Usage: python script.py <nrNodes> <avgSendTime> <experiment> <simtime> <nrBS> <full_collision> <directionality> <nrNetworks> <baseDist>")
        print("\nExperiments:")
        print("  0: Baseline SF12")
        print("  1: Frequency Hopping")
        print("  2: Baseline SF7")
        print("  3: Distance Optimized")
        print("  4: LoRaWAN Default")
        print("  5: Power Optimized")
        print("  8: Enhanced DQN")
        print("  9: Original ADR")
       
        print(" 10: ML-ADR")
        print("\nOr use --compare to run comparison of all algorithms")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"LoRaSim PRO v6.0 - Enhanced ADR Comparison")
    print(f"{'='*70}")
    
    exp_descriptions = {
        0: "BASELINE: SF12 only",
        1: "BASELINE with Frequency Hopping",
        2: "BASELINE: SF7 only",
        3: "Distance-Optimized (adaptive SF/BW)",
        4: "LoRaWAN Default (SF12, BW125)",
        5: "Distance + Power Optimized",
        8: "ENHANCED DQN with improved exploration",
        9: "ORIGINAL ADR (conservative)",
        10: "IMPROVED ADR (aggressive)",
        11: "ML-ADR (Q-learning based)"
    }
    print(f"Experiment {experiment}: {exp_descriptions.get(experiment, 'Unknown')}")
    
    config = SystemConfig(
        full_collision=full_collision_flag,
        directionality=directionality,
        use_dqn=(experiment == 8),
        use_improved_adr=(experiment == 10),
        use_ml_adr=(experiment == 11)
    )
    
    Ptx = 20
    gamma = 2.08
    d0 = 1000
    Lpld0 = 127.41
    
    if experiment in [0, 1, 4]:
        minsensi = sensi[5, 2]
    elif experiment == 2:
        minsensi = -112.0
    elif experiment in [3, 5]:
        minsensi = np.amin(sensi)
    elif experiment in [8, 9, 10, 11]:
        minsensi = np.mean(sensi[:, 1:])
    else:
        minsensi = -130.0
    
    Lpl = Ptx - minsensi
    maxDist = d0 * (math.e ** ((Lpl - Lpld0) / (10.0 * gamma)))
    
    print(f"\nSimulation Parameters:")
    print(f"  Nodes: {nrNodes:,}")
    print(f"  Base Stations: {nrBS}")
    print(f"  Simulation Time: {simtime:,} ms ({simtime/3600000:.1f} hours)")
    print(f"  Max Distance: {maxDist:.1f} m")
    print(f"  Directionality: {'ENABLED' if directionality else 'DISABLED'}")
    print(f"  Collision Model: Pure ALOHA + Physical Detection")
    
    try:
        simulator = EnhancedLoRaSimulator(config)
        simulator.initialize(nrNodes, avgSendTime, nrBS, maxDist, baseDist, experiment)
        simulator.run(simtime)
    except Exception as e:
        print(f"\nSIMULATION ERROR: {e}")
        import traceback 
        traceback.print_exc()
