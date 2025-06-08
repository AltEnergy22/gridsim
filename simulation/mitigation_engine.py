"""
Mitigation Engine

Generates and executes mitigation action plans for power system violations.
Implements rule-based strategies with economic optimization.
"""

import json
import copy
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandapower as pp
import numpy as np
from loguru import logger

from grid.advanced_grid import AdvancedGrid
from simulation.contingency_analyzer import ContingencyResult, ViolationDetails
from simulation.power_flow import PowerFlowEngine


@dataclass
class ActionStep:
    """Individual step in a mitigation action sequence"""
    step: int
    type: str  # 'redispatch', 'load_shed', 'tap_change', 'facts_adjust', 'topology_change'
    target: str  # equipment ID
    parameters: Dict[str, Any]
    timeout_s: float = 60.0
    condition: Optional[Dict[str, Any]] = None  # conditional execution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'step': self.step,
            'type': self.type,
            'target': self.target,
            'parameters': self.parameters,
            'timeout_s': self.timeout_s,
            'condition': self.condition
        }


@dataclass
class MitigationPlan:
    """Complete mitigation action plan"""
    action_id: str
    contingency_id: str
    created_by: str = "MitigationEngine"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Pre-conditions
    pre_conditions: Dict[str, str] = field(default_factory=dict)
    
    # Action sequence
    action_sequence: List[ActionStep] = field(default_factory=list)
    
    # Expected outcomes
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    
    # Cost breakdown
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Execution profile
    execution_profile: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Export plan to JSON"""
        plan_dict = {
            'action_id': self.action_id,
            'contingency_id': self.contingency_id,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'pre_conditions': self.pre_conditions,
            'action_sequence': [step.to_dict() for step in self.action_sequence],
            'expected_outcome': self.expected_outcome,
            'cost_breakdown': self.cost_breakdown,
            'execution_profile': self.execution_profile
        }
        return json.dumps(plan_dict, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MitigationPlan':
        """Load plan from JSON"""
        data = json.loads(json_str)
        
        # Convert action sequence
        action_sequence = []
        for step_data in data.get('action_sequence', []):
            step = ActionStep(**step_data)
            action_sequence.append(step)
        
        # Create plan
        plan = cls(
            action_id=data['action_id'],
            contingency_id=data['contingency_id'],
            created_by=data.get('created_by', 'Unknown'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            pre_conditions=data.get('pre_conditions', {}),
            action_sequence=action_sequence,
            expected_outcome=data.get('expected_outcome', {}),
            cost_breakdown=data.get('cost_breakdown', {}),
            execution_profile=data.get('execution_profile', {})
        )
        
        return plan


@dataclass
class ExecutionResult:
    """Results from executing a mitigation plan"""
    plan: MitigationPlan
    executed_at: str
    success: bool
    execution_time_s: float
    step_results: List[Dict[str, Any]]
    
    # Post-execution state
    violations_cleared: int
    violations_remaining: int
    final_violations: List[ViolationDetails]
    
    # Final metrics
    final_cost_usd: float
    total_mw_redispatched: float
    total_mw_shed: float
    
    # Performance
    convergence_time_s: float
    final_max_voltage_pu: float
    final_min_voltage_pu: float
    final_max_loading_pct: float


class MitigationEngine:
    """
    Comprehensive mitigation planning and execution engine
    
    Generates rule-based mitigation plans for power system violations
    and executes them with economic optimization.
    """
    
    def __init__(self, grid: AdvancedGrid):
        self.grid = grid
        self.power_flow_engine = PowerFlowEngine()
        
        # Economic parameters
        self.fuel_costs = {
            'coal': 2.5,    # $/MMBtu
            'gas': 4.0,
            'nuclear': 0.5,
            'hydro': 0.0,
            'wind': 0.0,
            'solar': 0.0,
            'oil': 8.0
        }
        
        self.voLL = 10000.0  # Value of lost load $/MWh
        
        # Heat rates (MMBtu/MWh)
        self.heat_rates = {
            'coal': 10.5,
            'gas': 9.0,
            'nuclear': 10.4,
            'hydro': 0.0,
            'wind': 0.0,
            'solar': 0.0,
            'oil': 11.0
        }
        
        # Ramp rates (MW/min)
        self.default_ramp_rates = {
            'coal': 3.0,
            'gas': 10.0,
            'nuclear': 2.0,
            'hydro': 100.0,
            'wind': 50.0,
            'solar': 50.0,
            'oil': 15.0
        }
    
    def generate_mitigation_plan(self, 
                               contingency_result: ContingencyResult,
                               base_net: pp.pandapowerNet) -> Optional[MitigationPlan]:
        """
        Generate a mitigation plan for a contingency result
        
        Args:
            contingency_result: Results from contingency analysis
            base_net: Base pandapower network
            
        Returns:
            Mitigation plan or None if no action needed
        """
        
        # Skip if no new violations
        if not contingency_result.new_violations:
            return None
        
        logger.info(f"Generating mitigation plan for contingency {contingency_result.contingency.id}")
        
        # Create plan ID
        action_id = f"ACT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{contingency_result.contingency.id}"
        
        plan = MitigationPlan(
            action_id=action_id,
            contingency_id=contingency_result.contingency.id
        )
        
        # Set pre-conditions
        plan.pre_conditions = {
            'min_spinning_reserve_mw': '50.0',
            'max_line_loading_pct': '120.0',
            'voltage_range_pu': '[0.90, 1.10]'
        }
        
        # Generate action sequence based on violations
        step_counter = 1
        total_cost = 0.0
        
        # Group violations by type for efficient handling
        voltage_violations = [v for v in contingency_result.new_violations 
                            if 'voltage' in v.violation_type]
        thermal_violations = [v for v in contingency_result.new_violations 
                            if 'thermal' in v.violation_type]
        
        # Handle thermal violations first (usually more critical)
        for violation in thermal_violations:
            step, cost = self._create_thermal_mitigation_step(violation, step_counter, base_net)
            if step:
                plan.action_sequence.append(step)
                total_cost += cost
                step_counter += 1
        
        # Handle voltage violations
        for violation in voltage_violations:
            step, cost = self._create_voltage_mitigation_step(violation, step_counter, base_net)
            if step:
                plan.action_sequence.append(step)
                total_cost += cost
                step_counter += 1
        
        # Set expected outcomes
        plan.expected_outcome = {
            'metrics': {
                'max_loading_pct': '<95',
                'min_voltage_pu': '>=0.98',
                'max_voltage_pu': '<=1.02'
            },
            'tolerance': {
                'voltage_deadband_pu': 0.02,
                'loading_deadband_pct': 2.0
            }
        }
        
        # Calculate costs
        plan.cost_breakdown = self._calculate_plan_costs(plan, base_net)
        
        # Set execution profile
        plan.execution_profile = {
            'estimated_time_s': len(plan.action_sequence) * 30.0,  # 30s per step estimate
            'priority': self._assess_plan_priority(contingency_result),
            'automation_level': 'semi_automatic'  # Requires operator approval
        }
        
        return plan
    
    def _create_thermal_mitigation_step(self, 
                                      violation: ViolationDetails,
                                      step_num: int,
                                      base_net: pp.pandapowerNet) -> Tuple[Optional[ActionStep], float]:
        """Create mitigation step for thermal violations"""
        
        overload_mw = violation.severity * violation.actual_value / 100.0  # Rough MW estimate
        
        # Strategy 1: Generator redispatch near the overloaded element
        if overload_mw < 100:  # Small overloads
            target_gen = self._find_nearby_generator(violation.element_id, base_net)
            if target_gen:
                # Increase generation upstream to reduce flow
                delta_mw = min(50.0, overload_mw * 1.2)  # 20% margin
                
                step = ActionStep(
                    step=step_num,
                    type='redispatch',
                    target=target_gen,
                    parameters={
                        'delta_mw': delta_mw,
                        'ramp_rate_mw_per_min': 10.0,
                        'reason': f'Reduce loading on {violation.element_id}'
                    }
                )
                
                cost = self._calculate_redispatch_cost(target_gen, delta_mw, base_net)
                return step, cost
        
        # Strategy 2: Load shedding for larger overloads
        else:
            target_bus = self._find_load_shedding_target(violation.element_id, base_net)
            if target_bus:
                shed_mw = min(20.0, overload_mw * 0.8)  # Conservative shedding
                
                step = ActionStep(
                    step=step_num,
                    type='load_shed',
                    target=target_bus,
                    parameters={
                        'shed_mw': shed_mw,
                        'shed_priority': 'interruptible',
                        'reason': f'Relieve overload on {violation.element_id}'
                    }
                )
                
                cost = shed_mw * self.voLL
                return step, cost
        
        return None, 0.0
    
    def _create_voltage_mitigation_step(self,
                                      violation: ViolationDetails,
                                      step_num: int,
                                      base_net: pp.pandapowerNet) -> Tuple[Optional[ActionStep], float]:
        """Create mitigation step for voltage violations"""
        
        if violation.violation_type == 'voltage_low':
            # Low voltage: increase reactive power or tap up transformers
            target_gen = self._find_voltage_controlling_generator(violation.element_id, base_net)
            if target_gen:
                # Increase reactive power output
                delta_q = min(50.0, violation.severity * 100)  # MVAR
                
                step = ActionStep(
                    step=step_num,
                    type='reactive_dispatch',
                    target=target_gen,
                    parameters={
                        'delta_mvar': delta_q,
                        'reason': f'Raise voltage at {violation.element_id}'
                    }
                )
                
                return step, 5.0  # Minimal cost for reactive power
        
        elif violation.violation_type == 'voltage_high':
            # High voltage: decrease reactive power or tap down transformers
            target_gen = self._find_voltage_controlling_generator(violation.element_id, base_net)
            if target_gen:
                # Decrease reactive power output
                delta_q = -min(50.0, violation.severity * 100)  # Negative MVAR
                
                step = ActionStep(
                    step=step_num,
                    type='reactive_dispatch', 
                    target=target_gen,
                    parameters={
                        'delta_mvar': delta_q,
                        'reason': f'Lower voltage at {violation.element_id}'
                    }
                )
                
                return step, 5.0  # Minimal cost for reactive power
        
        return None, 0.0
    
    def execute_mitigation_plan(self,
                              plan: MitigationPlan,
                              base_net: pp.pandapowerNet) -> ExecutionResult:
        """
        Execute a mitigation plan on a power system network
        
        Args:
            plan: Mitigation plan to execute
            base_net: Base pandapower network
            
        Returns:
            Execution results with metrics
        """
        
        logger.info(f"Executing mitigation plan {plan.action_id}")
        
        start_time = time.time()
        net = copy.deepcopy(base_net)
        
        step_results = []
        success = True
        total_redispatch = 0.0
        total_shed = 0.0
        
        # Execute each step in sequence
        for step in plan.action_sequence:
            step_start = time.time()
            
            try:
                # Execute the step
                step_success, step_metrics = self._execute_action_step(step, net)
                
                step_time = time.time() - step_start
                
                step_result = {
                    'step': step.step,
                    'type': step.type,
                    'target': step.target,
                    'success': step_success,
                    'execution_time_s': step_time,
                    'metrics': step_metrics
                }
                
                step_results.append(step_result)
                
                # Track totals
                if step.type == 'redispatch':
                    total_redispatch += step.parameters.get('delta_mw', 0)
                elif step.type == 'load_shed':
                    total_shed += step.parameters.get('shed_mw', 0)
                
                if not step_success:
                    logger.warning(f"Step {step.step} failed in plan {plan.action_id}")
                    success = False
                    break
                    
            except Exception as e:
                logger.error(f"Error executing step {step.step}: {e}")
                step_results.append({
                    'step': step.step,
                    'success': False,
                    'error': str(e),
                    'execution_time_s': time.time() - step_start
                })
                success = False
                break
        
        # Run final power flow to assess results
        convergence_start = time.time()
        try:
            pf_result = self.power_flow_engine.solve_power_flow(net)
            convergence_time = time.time() - convergence_start
            
            if pf_result.converged:
                # Count remaining violations
                final_violations = self._detect_final_violations(net)
                violations_remaining = len(final_violations)
                violations_cleared = len(plan.action_sequence) - violations_remaining  # Rough estimate
                
                final_metrics = {
                    'max_voltage_pu': pf_result.max_voltage_pu,
                    'min_voltage_pu': pf_result.min_voltage_pu,
                    'max_loading_pct': pf_result.max_line_loading_pct
                }
            else:
                logger.error("Final power flow did not converge")
                success = False
                final_violations = []
                violations_remaining = 999
                violations_cleared = 0
                final_metrics = {'max_voltage_pu': 0, 'min_voltage_pu': 0, 'max_loading_pct': 999}
                
        except Exception as e:
            logger.error(f"Error in final power flow: {e}")
            success = False
            convergence_time = 0
            final_violations = []
            violations_remaining = 999
            violations_cleared = 0
            final_metrics = {'max_voltage_pu': 0, 'min_voltage_pu': 0, 'max_loading_pct': 999}
        
        execution_time = time.time() - start_time
        
        # Calculate final cost
        final_cost = sum(plan.cost_breakdown.values())
        
        result = ExecutionResult(
            plan=plan,
            executed_at=datetime.now().isoformat(),
            success=success,
            execution_time_s=execution_time,
            step_results=step_results,
            violations_cleared=violations_cleared,
            violations_remaining=violations_remaining,
            final_violations=final_violations,
            final_cost_usd=final_cost,
            total_mw_redispatched=total_redispatch,
            total_mw_shed=total_shed,
            convergence_time_s=convergence_time,
            final_max_voltage_pu=final_metrics.get('max_voltage_pu', 0),
            final_min_voltage_pu=final_metrics.get('min_voltage_pu', 0),
            final_max_loading_pct=final_metrics.get('max_loading_pct', 0)
        )
        
        return result
    
    def _execute_action_step(self, step: ActionStep, net: pp.pandapowerNet) -> Tuple[bool, Dict[str, Any]]:
        """Execute a single action step"""
        
        try:
            if step.type == 'redispatch':
                return self._execute_redispatch(step, net)
            elif step.type == 'load_shed':
                return self._execute_load_shed(step, net)
            elif step.type == 'reactive_dispatch':
                return self._execute_reactive_dispatch(step, net)
            elif step.type == 'tap_change':
                return self._execute_tap_change(step, net)
            else:
                logger.warning(f"Unknown action type: {step.type}")
                return False, {'error': f'Unknown action type: {step.type}'}
                
        except Exception as e:
            logger.error(f"Error executing {step.type} on {step.target}: {e}")
            return False, {'error': str(e)}
    
    def _execute_redispatch(self, step: ActionStep, net: pp.pandapowerNet) -> Tuple[bool, Dict[str, Any]]:
        """Execute generator redispatch"""
        
        delta_mw = step.parameters.get('delta_mw', 0)
        target_gen = step.target
        
        # Find generator in network
        gen_mask = net.gen['name'] == target_gen
        if not gen_mask.any():
            return False, {'error': f'Generator {target_gen} not found'}
        
        gen_idx = net.gen.index[gen_mask][0]
        current_output = net.gen.at[gen_idx, 'p_mw']
        max_output = net.gen.at[gen_idx, 'max_p_mw']
        
        new_output = current_output + delta_mw
        
        # Check limits
        if new_output > max_output:
            new_output = max_output
            delta_mw = new_output - current_output
        elif new_output < 0:
            new_output = 0
            delta_mw = new_output - current_output
        
        # Apply redispatch
        net.gen.at[gen_idx, 'p_mw'] = new_output
        
        return True, {
            'delta_mw_actual': delta_mw,
            'new_output_mw': new_output,
            'previous_output_mw': current_output
        }
    
    def _execute_load_shed(self, step: ActionStep, net: pp.pandapowerNet) -> Tuple[bool, Dict[str, Any]]:
        """Execute load shedding"""
        
        shed_mw = step.parameters.get('shed_mw', 0)
        target_bus = step.target
        
        # Find loads at target bus
        load_mask = net.load['bus'].isin(net.bus.index[net.bus['name'] == target_bus])
        if not load_mask.any():
            return False, {'error': f'No loads found at bus {target_bus}'}
        
        load_indices = net.load.index[load_mask]
        total_load = net.load.loc[load_indices, 'p_mw'].sum()
        
        if total_load == 0:
            return False, {'error': f'No load to shed at bus {target_bus}'}
        
        # Proportionally reduce loads
        shed_factor = min(1.0, shed_mw / total_load)
        actual_shed = 0.0
        
        for idx in load_indices:
            original_load = net.load.at[idx, 'p_mw']
            reduction = original_load * shed_factor
            net.load.at[idx, 'p_mw'] = original_load - reduction
            actual_shed += reduction
        
        return True, {
            'shed_mw_requested': shed_mw,
            'shed_mw_actual': actual_shed,
            'total_load_before_mw': total_load,
            'shed_factor': shed_factor
        }
    
    def _execute_reactive_dispatch(self, step: ActionStep, net: pp.pandapowerNet) -> Tuple[bool, Dict[str, Any]]:
        """Execute reactive power dispatch"""
        
        delta_mvar = step.parameters.get('delta_mvar', 0)
        target_gen = step.target
        
        # Find generator in network
        gen_mask = net.gen['name'] == target_gen
        if not gen_mask.any():
            return False, {'error': f'Generator {target_gen} not found'}
        
        gen_idx = net.gen.index[gen_mask][0]
        
        # For PV buses, this would adjust voltage setpoint
        # For PQ buses, adjust reactive power directly
        if 'vm_pu' in net.gen.columns and not pd.isna(net.gen.at[gen_idx, 'vm_pu']):
            # PV bus - adjust voltage setpoint
            current_vm = net.gen.at[gen_idx, 'vm_pu']
            voltage_adjustment = delta_mvar * 0.001  # Simple conversion
            new_vm = max(0.95, min(1.05, current_vm + voltage_adjustment))
            net.gen.at[gen_idx, 'vm_pu'] = new_vm
            
            return True, {
                'voltage_adjustment_pu': voltage_adjustment,
                'new_voltage_setpoint_pu': new_vm,
                'previous_voltage_setpoint_pu': current_vm
            }
        else:
            # PQ bus - adjust reactive power
            current_q = net.gen.at[gen_idx, 'q_mvar'] if 'q_mvar' in net.gen.columns else 0
            new_q = current_q + delta_mvar
            
            # Apply reactive power limits if available
            max_q = net.gen.at[gen_idx, 'max_q_mvar'] if 'max_q_mvar' in net.gen.columns else 999
            min_q = net.gen.at[gen_idx, 'min_q_mvar'] if 'min_q_mvar' in net.gen.columns else -999
            
            new_q = max(min_q, min(max_q, new_q))
            net.gen.at[gen_idx, 'q_mvar'] = new_q
            
            return True, {
                'delta_mvar_actual': new_q - current_q,
                'new_q_mvar': new_q,
                'previous_q_mvar': current_q
            }
    
    def _execute_tap_change(self, step: ActionStep, net: pp.pandapowerNet) -> Tuple[bool, Dict[str, Any]]:
        """Execute transformer tap change"""
        
        tap_step = step.parameters.get('tap_step', 0)
        target_xfmr = step.target
        
        # Find transformer in network
        trafo_mask = net.trafo['name'] == target_xfmr
        if not trafo_mask.any():
            return False, {'error': f'Transformer {target_xfmr} not found'}
        
        trafo_idx = net.trafo.index[trafo_mask][0]
        
        # Apply tap change (simplified)
        if 'tap_pos' in net.trafo.columns:
            current_tap = net.trafo.at[trafo_idx, 'tap_pos']
            new_tap = current_tap + tap_step
            net.trafo.at[trafo_idx, 'tap_pos'] = new_tap
            
            return True, {
                'tap_step_actual': tap_step,
                'new_tap_position': new_tap,
                'previous_tap_position': current_tap
            }
        else:
            return False, {'error': 'Transformer does not support tap changing'}
    
    def _find_nearby_generator(self, element_id: str, net: pp.pandapowerNet) -> Optional[str]:
        """Find a generator near the given element for redispatch"""
        # Simplified: return first available generator with capacity
        for idx, gen in net.gen.iterrows():
            if gen['in_service'] and gen['p_mw'] < gen['max_p_mw'] * 0.9:
                return gen['name']
        return None
    
    def _find_load_shedding_target(self, element_id: str, net: pp.pandapowerNet) -> Optional[str]:
        """Find appropriate load for shedding near the overloaded element"""
        # Simplified: return bus with largest load
        if len(net.load) > 0:
            max_load_idx = net.load['p_mw'].idxmax()
            bus_idx = net.load.at[max_load_idx, 'bus']
            return net.bus.at[bus_idx, 'name']
        return None
    
    def _find_voltage_controlling_generator(self, bus_id: str, net: pp.pandapowerNet) -> Optional[str]:
        """Find voltage controlling generator near the given bus"""
        # Simplified: return first PV generator
        for idx, gen in net.gen.iterrows():
            if gen['in_service'] and 'vm_pu' in net.gen.columns and not pd.isna(gen['vm_pu']):
                return gen['name']
        return None
    
    def _calculate_plan_costs(self, plan: MitigationPlan, net: pp.pandapowerNet) -> Dict[str, float]:
        """Calculate detailed cost breakdown for mitigation plan"""
        
        costs = {
            'fuel_cost_usd': 0.0,
            'shed_cost_usd': 0.0,
            'startup_cost_usd': 0.0,
            'wear_tear_cost_usd': 0.0,
            'total_cost_usd': 0.0
        }
        
        for step in plan.action_sequence:
            if step.type == 'redispatch':
                delta_mw = step.parameters.get('delta_mw', 0)
                fuel_cost = self._calculate_redispatch_cost(step.target, delta_mw, net)
                costs['fuel_cost_usd'] += fuel_cost
                
            elif step.type == 'load_shed':
                shed_mw = step.parameters.get('shed_mw', 0)
                shed_cost = shed_mw * self.voLL
                costs['shed_cost_usd'] += shed_cost
                
            elif step.type in ['tap_change', 'reactive_dispatch']:
                costs['wear_tear_cost_usd'] += 10.0  # Minimal equipment wear cost
        
        costs['total_cost_usd'] = sum([v for k, v in costs.items() if k != 'total_cost_usd'])
        
        return costs
    
    def _calculate_redispatch_cost(self, gen_name: str, delta_mw: float, net: pp.pandapowerNet) -> float:
        """Calculate cost of generator redispatch"""
        
        # Find generator type from name (simplified)
        gen_type = 'gas'  # Default
        for fuel in self.fuel_costs.keys():
            if fuel in gen_name.lower():
                gen_type = fuel
                break
        
        # Calculate incremental cost
        heat_rate = self.heat_rates.get(gen_type, 9.0)
        fuel_cost = self.fuel_costs.get(gen_type, 4.0)
        
        incremental_cost = heat_rate * fuel_cost  # $/MWh
        
        return abs(delta_mw) * incremental_cost
    
    def _assess_plan_priority(self, contingency_result: ContingencyResult) -> str:
        """Assess priority level for mitigation plan"""
        
        if contingency_result.criticality_level == 'critical':
            return 'emergency'
        elif contingency_result.criticality_level == 'high':
            return 'high'
        elif contingency_result.criticality_level == 'medium':
            return 'medium'
        else:
            return 'low'
    
    def _detect_final_violations(self, net: pp.pandapowerNet) -> List[ViolationDetails]:
        """Detect violations after mitigation execution"""
        
        violations = []
        
        # Simplified violation detection (reuse analyzer logic)
        # This should be consistent with ContingencyAnalyzer
        
        # Voltage violations
        if hasattr(net, 'res_bus'):
            for idx, row in net.res_bus.iterrows():
                voltage_pu = row['vm_pu']
                bus_name = net.bus.loc[idx, 'name']
                
                if voltage_pu < 0.95 or voltage_pu > 1.05:
                    violation_type = 'voltage_low' if voltage_pu < 0.95 else 'voltage_high'
                    severity = abs(voltage_pu - (0.95 if voltage_pu < 0.95 else 1.05))
                    
                    violations.append(ViolationDetails(
                        element_id=bus_name,
                        element_name=bus_name,
                        element_type='bus',
                        violation_type=violation_type,
                        severity=severity,
                        limit=0.95 if voltage_pu < 0.95 else 1.05,
                        actual_value=voltage_pu,
                        critical=voltage_pu < 0.90 or voltage_pu > 1.10,
                        location=bus_name
                    ))
        
        # Line thermal violations
        if hasattr(net, 'res_line'):
            for idx, row in net.res_line.iterrows():
                loading_pct = row['loading_percent']
                if loading_pct > 100.0:
                    line_name = net.line.loc[idx, 'name']
                    violations.append(ViolationDetails(
                        element_id=line_name,
                        element_name=line_name,
                        element_type='line',
                        violation_type='thermal_overload',
                        severity=loading_pct - 100.0,
                        limit=100.0,
                        actual_value=loading_pct,
                        critical=loading_pct > 150.0,
                        location=f"{net.line.loc[idx, 'from_bus']}-{net.line.loc[idx, 'to_bus']}"
                    ))
        
        return violations


def save_mitigation_plans(plans: List[MitigationPlan], output_dir: str = "outputs/mitigation_plans"):
    """Save mitigation plans to JSON files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plans_index = []
    
    for plan in plans:
        # Save individual plan
        plan_file = output_path / f"{plan.action_id}.json"
        with open(plan_file, 'w') as f:
            f.write(plan.to_json())
        
        # Add to index
        plans_index.append({
            'action_id': plan.action_id,
            'contingency_id': plan.contingency_id,
            'created_at': plan.created_at,
            'file': f"{plan.action_id}.json"
        })
    
    # Save index
    index_file = output_path / "plans_index.json"
    with open(index_file, 'w') as f:
        json.dump(plans_index, f, indent=2)
    
    logger.info(f"Saved {len(plans)} mitigation plans to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from grid.advanced_grid import AdvancedGrid
    from simulation.build_network import build_pandapower_network
    from simulation.contingency_generator import load_contingencies
    from simulation.contingency_analyzer import ContingencyAnalyzer
    
    # Create test setup
    regions = ['A', 'B', 'C']
    grid = AdvancedGrid(regions, buses_per_region=50)
    base_net, _ = build_pandapower_network(grid)
    
    # Load contingencies and analyze a few
    contingencies = load_contingencies("data/contingencies")[:5]
    
    if contingencies:
        analyzer = ContingencyAnalyzer(grid)
        results = analyzer.analyze_all_contingencies(base_net, contingencies)
        
        # Generate mitigation plans
        engine = MitigationEngine(grid)
        plans = []
        
        for result in results:
            if result.new_violations:
                plan = engine.generate_mitigation_plan(result, base_net)
                if plan:
                    plans.append(plan)
                    
                    # Test execution
                    execution_result = engine.execute_mitigation_plan(plan, base_net)
                    print(f"Plan {plan.action_id}: Success={execution_result.success}, "
                          f"Cost=${execution_result.final_cost_usd:.2f}")
        
        if plans:
            save_mitigation_plans(plans)
            print(f"Generated and saved {len(plans)} mitigation plans")
    else:
        print("No contingencies found for testing") 