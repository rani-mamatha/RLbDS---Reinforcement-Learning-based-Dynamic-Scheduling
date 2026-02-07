"""
Workload Generator for simulating IoT tasks
Based on Bitbrain dataset characteristics
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import random


class Task:
    """Task class representing a single task"""
    
    def __init__(self, task_id, cpu_required, ram_required, bandwidth_required, 
                 deadline, priority, arrival_time, duration):
        self.task_id = task_id
        self.cpu_required = cpu_required
        self.ram_required = ram_required
        self.bandwidth_required = bandwidth_required
        self.deadline = deadline
        self.priority = priority
        self.arrival_time = arrival_time
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.assigned_host = None
        self.is_completed = False
        self.is_migrated = False
        
    def to_dict(self):
        """Convert task to dictionary"""
        return {
            'task_id': self.task_id,
            'cpu_required': self.cpu_required,
            'ram_required': self.ram_required,
            'bandwidth_required': self.bandwidth_required,
            'deadline': self.deadline,
            'priority': self.priority,
            'arrival_time': self.arrival_time,
            'duration': self.duration,
            'assigned_host': self.assigned_host,
            'is_completed': self.is_completed
        }


class WorkloadGenerator:
    """Generate dynamic workload based on Bitbrain traces"""
    
    def __init__(self, config, dataset_path=None):
        self.config = config
        self.dataset_path = dataset_path
        self.task_counter = 0
        
        # Load real traces if available
        self.real_traces = None
        if dataset_path:
            try:
                self.real_traces = self._load_bitbrain_traces(dataset_path)
            except:
                print("Warning: Could not load Bitbrain traces. Using synthetic workload.")
        
    def _load_bitbrain_traces(self, path):
        """Load Bitbrain dataset traces"""
        try:
            # Assuming CSV format with columns: timestamp, cpu, ram, bandwidth
            df = pd.read_csv(path)
            return df
        except Exception as e:
            print(f"Error loading traces: {e}")
            return None
    
    def generate_tasks_for_interval(self, interval_num, num_tasks=None):
        """
        Generate tasks for a specific scheduling interval
        Args:
            interval_num: Current scheduling interval number
            num_tasks: Number of tasks to generate (random if None)
        Returns:
            List of Task objects
        """
        if num_tasks is None:
            # Random number of tasks between 10 and 50
            num_tasks = random.randint(10, 50)
        
        tasks = []
        interval_duration = self.config['environment']['interval_duration']
        
        for i in range(num_tasks):
            task = self._generate_single_task(interval_num, interval_duration)
            tasks.append(task)
        
        return tasks
    
    def _generate_single_task(self, interval_num, interval_duration):
        """Generate a single task with random or trace-based characteristics"""
        
        self.task_counter += 1
        
        if self.real_traces is not None and len(self.real_traces) > self.task_counter:
            # Use real trace data
            trace = self.real_traces.iloc[self.task_counter % len(self.real_traces)]
            cpu_required = trace.get('cpu', random.uniform(0.5, 8.0))
            ram_required = trace.get('ram', random.uniform(1.0, 16.0))
            bandwidth_required = trace.get('bandwidth', random.uniform(5.0, 100.0))
        else:
            # Generate synthetic task
            cpu_required = random.uniform(0.5, 8.0)  # CPU cores
            ram_required = random.uniform(1.0, 16.0)  # GB
            bandwidth_required = random.uniform(5.0, 100.0)  # Mbps
        
        # Task characteristics
        arrival_time = interval_num * interval_duration + random.uniform(0, interval_duration)
        duration = random.uniform(10, 300)  # 10 seconds to 5 minutes
        deadline = arrival_time + duration * random.uniform(1.5, 3.0)
        priority = random.randint(1, 5)  # 1 = highest, 5 = lowest
        
        task = Task(
            task_id=self.task_counter,
            cpu_required=cpu_required,
            ram_required=ram_required,
            bandwidth_required=bandwidth_required,
            deadline=deadline,
            priority=priority,
            arrival_time=arrival_time,
            duration=duration
        )
        
        return task
    
    def generate_workload_sequence(self, num_intervals):
        """
        Generate complete workload sequence for multiple intervals
        Args:
            num_intervals: Number of scheduling intervals
        Returns:
            Dictionary mapping interval_num to list of tasks
        """
        workload = {}
        
        for interval in range(num_intervals):
            tasks = self.generate_tasks_for_interval(interval)
            workload[interval] = tasks
        
        return workload
    
    def get_task_statistics(self, tasks):
        """Get statistics about generated tasks"""
        if not tasks:
            return {}
        
        cpu_reqs = [t.cpu_required for t in tasks]
        ram_reqs = [t.ram_required for t in tasks]
        bw_reqs = [t.bandwidth_required for t in tasks]
        
        stats = {
            'num_tasks': len(tasks),
            'avg_cpu': np.mean(cpu_reqs),
            'avg_ram': np.mean(ram_reqs),
            'avg_bandwidth': np.mean(bw_reqs),
            'max_cpu': np.max(cpu_reqs),
            'max_ram': np.max(ram_reqs),
            'max_bandwidth': np.max(bw_reqs)
        }
        
        return stats