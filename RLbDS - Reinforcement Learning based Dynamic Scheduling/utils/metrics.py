"""
Metrics Calculator for evaluating scheduling performance
Implements metrics from the paper: AEC, ART, AMT, Cost, SLA violations
"""

import numpy as np
from typing import List, Dict


class MetricsCalculator:
    """Calculate performance metrics for task scheduling"""
    
    def __init__(self, config):
        self.config = config
        
        # Pricing parameters (Azure-based)
        self.edge_cost_per_hour = 0.05  # USD per hour
        self.cloud_cost_per_hour = 0.15  # USD per hour
        self.edge_power_factor = 0.5  # Energy efficiency factor
        self.cloud_power_factor = 1.0
        
    def calculate_average_energy_consumption(self, hosts, tasks, interval_duration):
        """
        Calculate Average Energy Consumption (AEC)
        Formula from paper: Equation 5
        """
        total_energy = 0.0
        max_power = 0.0
        
        for host in hosts:
            # Power consumption during the interval
            power = self._calculate_host_power(host)
            
            # Energy factor based on deployment (edge or cloud)
            energy_factor = self.edge_power_factor if host['is_edge'] else self.cloud_power_factor
            
            # Energy = Power * Time * Factor
            energy = power * interval_duration * energy_factor
            total_energy += energy
            
            # Track max power for normalization
            max_power = max(max_power, host['max_power'])
        
        # Normalize by maximum possible energy
        max_energy = max_power * interval_duration * len(hosts)
        
        if max_energy > 0:
            normalized_aec = total_energy / max_energy
        else:
            normalized_aec = 0.0
        
        return normalized_aec, total_energy
    
    def _calculate_host_power(self, host):
        """Calculate power consumption of a host based on utilization"""
        # Linear power model: P = P_idle + (P_max - P_idle) * utilization
        p_idle = host.get('idle_power', 100)  # Watts
        p_max = host.get('max_power', 300)  # Watts
        
        cpu_util = 1.0 - (host['cpu_available'] / host['cpu_total'])
        ram_util = 1.0 - (host['ram_available'] / host['ram_total'])
        
        # Average utilization
        utilization = (cpu_util + ram_util) / 2.0
        
        power = p_idle + (p_max - p_idle) * utilization
        
        return power
    
    def calculate_average_response_time(self, tasks):
        """
        Calculate Average Response Time (ART)
        Formula from paper: Equation 6
        """
        if not tasks:
            return 0.0, 0.0
        
        response_times = []
        
        for task in tasks:
            if task.start_time is not None and task.end_time is not None:
                response_time = task.end_time - task.arrival_time
                response_times.append(response_time)
        
        if not response_times:
            return 0.0, 0.0
        
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        # Normalize
        if max_response_time > 0:
            normalized_art = avg_response_time / max_response_time
        else:
            normalized_art = 0.0
        
        return normalized_art, avg_response_time
    
    def calculate_average_migration_time(self, tasks):
        """
        Calculate Average Migration Time (AMT)
        Formula from paper: Equation 7
        """
        migration_times = []
        
        for task in tasks:
            if task.is_migrated and hasattr(task, 'migration_time'):
                migration_times.append(task.migration_time)
        
        if not migration_times:
            return 0.0, 0.0
        
        avg_migration_time = np.mean(migration_times)
        max_migration_time = np.max(migration_times)
        
        # Normalize
        if max_migration_time > 0:
            normalized_amt = avg_migration_time / max_migration_time
        else:
            normalized_amt = 0.0
        
        return normalized_amt, avg_migration_time
    
    def calculate_cost(self, hosts, interval_duration):
        """
        Calculate total cost
        Formula from paper: Equation 8
        """
        total_cost = 0.0
        
        # Convert interval duration to hours
        hours = interval_duration / 3600.0
        
        for host in hosts:
            if host['is_edge']:
                cost_per_hour = self.edge_cost_per_hour
            else:
                cost_per_hour = self.cloud_cost_per_hour
            
            # Cost based on utilization
            cpu_util = 1.0 - (host['cpu_available'] / host['cpu_total'])
            cost = cost_per_hour * hours * cpu_util
            
            total_cost += cost
        
        return total_cost
    
    def calculate_sla_violations(self, tasks):
        """
        Calculate Average SLA Violations
        Formula from paper: Equation 9
        """
        if not tasks:
            return 0.0, 0
        
        violations = 0
        completed_tasks = 0
        
        for task in tasks:
            if task.is_completed:
                completed_tasks += 1
                if task.end_time > task.deadline:
                    violations += 1
        
        if completed_tasks > 0:
            sla_violation_rate = violations / completed_tasks
        else:
            sla_violation_rate = 0.0
        
        return sla_violation_rate, violations
    
    def calculate_all_metrics(self, hosts, tasks, interval_duration):
        """Calculate all metrics at once"""
        # AEC
        norm_aec, total_energy = self.calculate_average_energy_consumption(
            hosts, tasks, interval_duration
        )
        
        # ART
        norm_art, avg_response_time = self.calculate_average_response_time(tasks)
        
        # AMT
        norm_amt, avg_migration_time = self.calculate_average_migration_time(tasks)
        
        # Cost
        total_cost = self.calculate_cost(hosts, interval_duration)
        
        # SLA Violations
        sla_violation_rate, num_violations = self.calculate_sla_violations(tasks)
        
        metrics = {
            'normalized_aec': norm_aec,
            'total_energy': total_energy,
            'normalized_art': norm_art,
            'avg_response_time': avg_response_time,
            'normalized_amt': norm_amt,
            'avg_migration_time': avg_migration_time,
            'total_cost': total_cost,
            'sla_violation_rate': sla_violation_rate,
            'num_sla_violations': num_violations,
            'num_completed_tasks': sum(1 for t in tasks if t.is_completed)
        }
        
        return metrics