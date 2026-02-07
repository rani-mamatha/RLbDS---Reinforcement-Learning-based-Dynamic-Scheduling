"""
Constraint Satisfaction Module
Validates scheduling decisions and computes penalties
"""

import numpy as np


class ConstraintSatisfaction:
    """
    Validates constraints and computes penalty
    Ensures valid task assignments and migrations
    """
    
    def __init__(self, config):
        self.config = config
    
    def validate_action(self, action, hosts, tasks):
        """
        Validate if the suggested action is valid
        
        Args:
            action: Task-host assignments (array of host indices)
            hosts: List of available hosts
            tasks: List of tasks to schedule
        
        Returns:
            is_valid: Boolean indicating if action is valid
            penalty: Penalty value (0 if valid, >0 otherwise)
            violations: List of constraint violations
        """
        violations = []
        penalty = 0.0
        
        # Check each task assignment
        for task_idx, host_idx in enumerate(action):
            if task_idx >= len(tasks):
                continue
                
            task = tasks[task_idx]
            
            # Validate host index
            if host_idx < 0 or host_idx >= len(hosts):
                violations.append(f"Task {task.task_id}: Invalid host index {host_idx}")
                penalty += 1.0
                continue
            
            host = hosts[host_idx]
            
            # Check resource constraints
            resource_violations = self._check_resource_constraints(task, host)
            if resource_violations:
                violations.extend(resource_violations)
                penalty += len(resource_violations) * 0.5
            
            # Check if host is in migration
            if hasattr(host, 'in_migration') and host.in_migration:
                violations.append(f"Task {task.task_id}: Host {host_idx} is in migration")
                penalty += 0.3
            
            # Check if host capacity is exceeded
            if self._is_host_overloaded(host):
                violations.append(f"Task {task.task_id}: Host {host_idx} is overloaded")
                penalty += 0.5
        
        is_valid = len(violations) == 0
        
        return is_valid, penalty, violations
    
    def _check_resource_constraints(self, task, host):
        """Check if host has enough resources for task"""
        violations = []
        
        # CPU constraint
        if task.cpu_required > host.cpu_available:
            violations.append(
                f"Task {task.task_id}: Insufficient CPU "
                f"(required: {task.cpu_required}, available: {host.cpu_available})"
            )
        
        # RAM constraint
        if task.ram_required > host.ram_available:
            violations.append(
                f"Task {task.task_id}: Insufficient RAM "
                f"(required: {task.ram_required}, available: {host.ram_available})"
            )
        
        # Bandwidth constraint
        if task.bandwidth_required > host.bandwidth:
            violations.append(
                f"Task {task.task_id}: Insufficient Bandwidth "
                f"(required: {task.bandwidth_required}, available: {host.bandwidth})"
            )
        
        return violations
    
    def _is_host_overloaded(self, host):
        """Check if host is overloaded"""
        cpu_utilization = 1.0 - (host.cpu_available / host.cpu_total)
        ram_utilization = 1.0 - (host.ram_available / host.ram_total)
        
        # Consider overloaded if utilization > 95%
        return cpu_utilization > 0.95 or ram_utilization > 0.95
    
    def suggest_alternative(self, task, hosts):
        """
        Suggest an alternative host if current assignment is invalid
        
        Args:
            task: Task to schedule
            hosts: List of available hosts
        
        Returns:
            host_idx: Index of suggested host, or None if no suitable host
        """
        suitable_hosts = []
        
        for idx, host in enumerate(hosts):
            # Check if host is in migration
            in_migration = hasattr(host, 'in_migration') and host.in_migration
            
            # Check resource availability
            if (task.cpu_required <= host.cpu_available and
                task.ram_required <= host.ram_available and
                task.bandwidth_required <= host.bandwidth and
                not in_migration and
                not self._is_host_overloaded(host)):
                
                # Calculate score (prefer less loaded hosts)
                cpu_util = 1.0 - (host.cpu_available / host.cpu_total)
                ram_util = 1.0 - (host.ram_available / host.ram_total)
                avg_util = (cpu_util + ram_util) / 2.0
                
                # Prefer edge nodes for latency-sensitive tasks
                edge_bonus = 0.2 if host.is_edge else 0.0
                
                score = (1.0 - avg_util) + edge_bonus
                suitable_hosts.append((idx, score))
        
        if suitable_hosts:
            # Return host with highest score
            suitable_hosts.sort(key=lambda x: x[1], reverse=True)
            return suitable_hosts[0][0]
        
        return None
    
    def compute_migration_penalty(self, task, from_host, to_host):
        """
        Compute penalty for migrating a task
        
        Args:
            task: Task to migrate
            from_host: Current host
            to_host: Target host
        
        Returns:
            migration_penalty: Penalty for this migration
        """
        # Migration cost based on task size and network distance
        data_size = task.ram_required  # Assuming RAM size as data size
        
        # Network distance penalty (edge-to-cloud is more expensive)
        if from_host.is_edge and not to_host.is_edge:
            distance_penalty = 0.5
        elif not from_host.is_edge and to_host.is_edge:
            distance_penalty = 0.3
        else:
            distance_penalty = 0.1
        
        # Total migration penalty
        migration_penalty = data_size * distance_penalty * 0.01
        
        return migration_penalty