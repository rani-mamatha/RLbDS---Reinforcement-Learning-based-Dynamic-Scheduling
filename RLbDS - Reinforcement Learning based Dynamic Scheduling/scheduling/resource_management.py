"""
Resource Management Module
Handles task allocation, migration, and resource monitoring
"""

import numpy as np
from typing import List, Dict
import copy


class Host:
    """Host class representing edge or cloud node"""
    
    def __init__(self, host_id, cpu_total, ram_total, bandwidth, is_edge=False):
        self.host_id = host_id
        self.cpu_total = cpu_total
        self.ram_total = ram_total
        self.bandwidth = bandwidth
        self.is_edge = is_edge
        
        # Available resources
        self.cpu_available = cpu_total
        self.ram_available = ram_total
        
        # Power characteristics
        self.idle_power = 100 if is_edge else 150  # Watts
        self.max_power = 250 if is_edge else 400   # Watts
        
        # Running tasks
        self.running_tasks = []
        
        # Migration status
        self.in_migration = False
        
    def to_dict(self):
        """Convert host to dictionary"""
        return {
            'host_id': self.host_id,
            'cpu_total': self.cpu_total,
            'ram_total': self.ram_total,
            'cpu_available': self.cpu_available,
            'ram_available': self.ram_available,
            'bandwidth': self.bandwidth,
            'is_edge': self.is_edge,
            'idle_power': self.idle_power,
            'max_power': self.max_power,
            'in_migration': self.in_migration,
            'num_running_tasks': len(self.running_tasks)
        }
    
    def allocate_task(self, task):
        """Allocate resources for a task"""
        if (task.cpu_required <= self.cpu_available and 
            task.ram_required <= self.ram_available):
            
            self.cpu_available -= task.cpu_required
            self.ram_available -= task.ram_required
            self.running_tasks.append(task.task_id)
            return True
        return False
    
    def deallocate_task(self, task):
        """Deallocate resources after task completion"""
        self.cpu_available += task.cpu_required
        self.ram_available += task.ram_required
        if task.task_id in self.running_tasks:
            self.running_tasks.remove(task.task_id)


class ResourceManagement:
    """
    Resource Management Module
    Manages hosts, task allocation, and migration
    """
    
    def __init__(self, config):
        self.config = config
        self.hosts = []
        self.current_time = 0.0
        
        # Initialize hosts
        self._initialize_hosts()
    
    def _initialize_hosts(self):
        """Initialize edge and cloud hosts"""
        num_edge = self.config['infrastructure']['edge_nodes']
        num_cloud = self.config['infrastructure']['cloud_nodes']
        
        edge_cpu_range = self.config['infrastructure']['edge_cpu_range']
        cloud_cpu_range = self.config['infrastructure']['cloud_cpu_range']
        edge_ram_range = self.config['infrastructure']['edge_ram_range']
        cloud_ram_range = self.config['infrastructure']['cloud_ram_range']
        edge_bw_range = self.config['infrastructure']['edge_bandwidth_range']
        cloud_bw_range = self.config['infrastructure']['cloud_bandwidth_range']
        
        host_id = 0
        
        # Create edge hosts
        for i in range(num_edge):
            cpu = np.random.uniform(edge_cpu_range[0], edge_cpu_range[1])
            ram = np.random.uniform(edge_ram_range[0], edge_ram_range[1])
            bw = np.random.uniform(edge_bw_range[0], edge_bw_range[1])
            
            host = Host(host_id, cpu, ram, bw, is_edge=True)
            self.hosts.append(host)
            host_id += 1
        
        # Create cloud hosts
        for i in range(num_cloud):
            cpu = np.random.uniform(cloud_cpu_range[0], cloud_cpu_range[1])
            ram = np.random.uniform(cloud_ram_range[0], cloud_ram_range[1])
            bw = np.random.uniform(cloud_bw_range[0], cloud_bw_range[1])
            
            host = Host(host_id, cpu, ram, bw, is_edge=False)
            self.hosts.append(host)
            host_id += 1
    
    def get_state(self, new_tasks, remaining_tasks):
        """
        Get current state for DRL model
        
        Args:
            new_tasks: Newly arrived tasks
            remaining_tasks: Tasks from previous interval
        
        Returns:
            state: Dictionary containing hosts and tasks info
        """
        state = {
            'hosts': [host.to_dict() for host in self.hosts],
            'new_tasks': [task.to_dict() for task in new_tasks],
            'remaining_tasks': [task.to_dict() for task in remaining_tasks]
        }
        
        return state
    
    def allocate_tasks(self, action, tasks):
        """
        Allocate tasks based on DRL action
        
        Args:
            action: Array of host indices for each task
            tasks: List of tasks to allocate
        
        Returns:
            allocated_tasks: List of successfully allocated tasks
            failed_tasks: List of tasks that failed to allocate
        """
        allocated_tasks = []
        failed_tasks = []
        
        for task_idx, host_idx in enumerate(action):
            if task_idx >= len(tasks):
                break
            
            task = tasks[task_idx]
            
            # Validate host index
            if host_idx < 0 or host_idx >= len(self.hosts):
                failed_tasks.append(task)
                continue
            
            host = self.hosts[host_idx]
            
            # Try to allocate
            if host.allocate_task(task):
                task.assigned_host = host_idx
                task.start_time = self.current_time
                allocated_tasks.append(task)
            else:
                failed_tasks.append(task)
        
        return allocated_tasks, failed_tasks
    
    def migrate_task(self, task, from_host_idx, to_host_idx):
        """
        Migrate a task from one host to another
        
        Args:
            task: Task to migrate
            from_host_idx: Current host index
            to_host_idx: Target host index
        
        Returns:
            success: Boolean indicating migration success
            migration_time: Time taken for migration
        """
        if from_host_idx == to_host_idx:
            return True, 0.0
        
        from_host = self.hosts[from_host_idx]
        to_host = self.hosts[to_host_idx]
        
        # Check if target host can accommodate the task
        if (task.cpu_required <= to_host.cpu_available and 
            task.ram_required <= to_host.ram_available):
            
            # Deallocate from source
            from_host.deallocate_task(task)
            
            # Allocate to target
            to_host.allocate_task(task)
            
            # Calculate migration time (based on data size and bandwidth)
            data_size = task.ram_required * 1024  # GB to MB
            effective_bw = min(from_host.bandwidth, to_host.bandwidth)
            migration_time = data_size / effective_bw if effective_bw > 0 else 10.0
            
            # Update task
            task.assigned_host = to_host_idx
            task.is_migrated = True
            task.migration_time = migration_time
            
            # Mark hosts as in migration temporarily
            from_host.in_migration = True
            to_host.in_migration = True
            
            return True, migration_time
        
        return False, 0.0
    
    def execute_tasks(self, tasks, interval_duration):
        """
        Execute tasks for the given interval
        
        Args:
            tasks: List of active tasks
            interval_duration: Duration of the interval
        
        Returns:
            completed_tasks: List of completed tasks
            active_tasks: List of still active tasks
        """
        completed_tasks = []
        active_tasks = []
        
        for task in tasks:
            if task.start_time is None:
                active_tasks.append(task)
                continue
            
            # Calculate elapsed time
            elapsed = self.current_time + interval_duration - task.start_time
            
            if elapsed >= task.duration:
                # Task completed
                task.end_time = task.start_time + task.duration
                task.is_completed = True
                
                # Deallocate resources
                if task.assigned_host is not None:
                    host = self.hosts[task.assigned_host]
                    host.deallocate_task(task)
                
                completed_tasks.append(task)
            else:
                # Task still running
                active_tasks.append(task)
        
        # Update current time
        self.current_time += interval_duration
        
        # Reset migration flags
        for host in self.hosts:
            host.in_migration = False
        
        return completed_tasks, active_tasks
    
    def get_host_statistics(self):
        """Get statistics about hosts"""
        total_cpu = sum(h.cpu_total for h in self.hosts)
        available_cpu = sum(h.cpu_available for h in self.hosts)
        total_ram = sum(h.ram_total for h in self.hosts)
        available_ram = sum(h.ram_available for h in self.hosts)
        
        stats = {
            'total_hosts': len(self.hosts),
            'edge_hosts': sum(1 for h in self.hosts if h.is_edge),
            'cloud_hosts': sum(1 for h in self.hosts if not h.is_edge),
            'cpu_utilization': 1.0 - (available_cpu / total_cpu) if total_cpu > 0 else 0.0,
            'ram_utilization': 1.0 - (available_ram / total_ram) if total_ram > 0 else 0.0,
            'total_running_tasks': sum(len(h.running_tasks) for h in self.hosts)
        }
        
        return stats
    
    def reset(self):
        """Reset resource management state"""
        self.current_time = 0.0
        self._initialize_hosts()