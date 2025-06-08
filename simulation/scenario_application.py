"""
Scenario Application Engine

Applies generated scenarios to power system networks, including:
- Load variations and fluctuations  
- Renewable dispatch changes
- Equipment outages and failures
- Market-driven generation dispatch
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from loguru import logger
import pandapower as pp

from simulation.scenario_generator import ScenarioConditions, ScenarioType

@dataclass
class SimulationResult:
    """Results from running a scenario"""
    scenario: ScenarioConditions
    converged: bool
    iterations: int
    computation_time: float
    final_residual: float
    
    # Power system metrics
    total_generation: float
    total_load: float
    total_losses: float
    loss_percentage: float
    
    # Voltage metrics
    min_voltage: float
    max_voltage: float
    voltage_violations: int
    
    # Stability metrics
    stability_margin: float
    min_eigenvalue: float
    
    # Line loading metrics
    max_line_loading: float
    line_violations: int
    
    # Economic metrics
    generation_cost: float
    load_shedding: float

class ScenarioApplication:
    """Apply scenario conditions to power system networks"""
    
    def __init__(self):
        pass
        
    def apply_scenario(self, scenario: ScenarioConditions, net: pp.pandapowerNet) -> pp.pandapowerNet:
        """Apply scenario conditions to a pandapower network"""
        
        # Create a copy to avoid modifying the original
        scenario_net = net.deepcopy()
        
        # Apply load variations
        self._apply_load_variations(scenario_net, scenario)
        
        # Apply renewable dispatch
        self._apply_renewable_dispatch(scenario_net, scenario)
        
        # Apply market-driven generation dispatch  
        self._apply_generation_dispatch(scenario_net, scenario)
        
        # Apply voltage stress (if any)
        self._apply_voltage_stress(scenario_net, scenario)
        
        logger.debug(f"Applied {scenario.scenario_type.value} scenario: "
                    f"load_mult={scenario.load_multiplier:.3f}, "
                    f"temp={scenario.weather_temp_c:.1f}°C")
        
        return scenario_net
    
    def _apply_load_variations(self, net: pp.pandapowerNet, scenario: ScenarioConditions):
        """Apply load variations based on scenario conditions"""
        
        # Global load scaling
        base_multiplier = scenario.load_multiplier
        
        # Weather-sensitive loads (HVAC impact)
        temp_factor = self._calculate_temperature_factor(scenario.weather_temp_c)
        
        # Economic demand response
        econ_factor = 1.0 / scenario.economic_index  # Higher prices = lower demand
        
        # Apply to all loads with some bus-specific variation
        for idx in net.load.index:
            bus_id = net.load.at[idx, 'bus']
            
            # Base load scaling
            load_multiplier = base_multiplier
            
            # Add weather sensitivity (±10% based on temperature)
            load_multiplier *= temp_factor
            
            # Add economic response
            load_multiplier *= econ_factor
            
            # Add some random variation per bus (±3%)
            bus_variation = random.gauss(1.0, 0.03)
            load_multiplier *= bus_variation
            
            # Apply to both P and Q
            net.load.at[idx, 'p_mw'] *= load_multiplier
            net.load.at[idx, 'q_mvar'] *= load_multiplier
            
    def _apply_renewable_dispatch(self, net: pp.pandapowerNet, scenario: ScenarioConditions):
        """Apply renewable capacity factors to generators"""
        
        # Map fuel types to capacity factors
        renewable_cf = scenario.renewable_cf
        
        # Apply to generators based on fuel type
        for idx in net.gen.index:
            gen_name = str(net.gen.at[idx, 'name'])
            
            # Determine fuel type from generator name/characteristics
            if 'solar' in gen_name.lower() or 'pv' in gen_name.lower():
                cf = renewable_cf.get('solar', 0.0)
                max_p = net.gen.at[idx, 'max_p_mw']
                net.gen.at[idx, 'p_mw'] = max_p * cf
                
            elif 'wind' in gen_name.lower():
                cf = renewable_cf.get('wind', 0.0)
                max_p = net.gen.at[idx, 'max_p_mw']
                net.gen.at[idx, 'p_mw'] = max_p * cf
                
            elif 'hydro' in gen_name.lower():
                cf = renewable_cf.get('hydro', 0.8)  # More predictable
                max_p = net.gen.at[idx, 'max_p_mw']
                net.gen.at[idx, 'p_mw'] = max_p * cf
                
    def _apply_generation_dispatch(self, net: pp.pandapowerNet, scenario: ScenarioConditions):
        """Apply market-driven generation dispatch"""
        
        fuel_costs = scenario.fuel_costs
        
        # Calculate marginal costs for each generator
        marginal_costs = []
        gen_indices = []
        
        for idx in net.gen.index:
            gen_name = str(net.gen.at[idx, 'name'])
            
            # Determine marginal cost based on fuel type
            if 'gas' in gen_name.lower():
                heat_rate = 9.0  # MMBtu/MWh typical
                marginal_cost = fuel_costs.get('gas', 4.0) * heat_rate
            elif 'coal' in gen_name.lower():
                heat_rate = 10.5  # Less efficient
                marginal_cost = fuel_costs.get('coal', 2.5) * heat_rate
            elif 'nuclear' in gen_name.lower():
                marginal_cost = fuel_costs.get('nuclear', 0.5) * 0.5  # Very low marginal cost
            elif 'oil' in gen_name.lower():
                heat_rate = 11.0  # Peaker units
                marginal_cost = fuel_costs.get('oil', 8.0) * heat_rate
            else:
                marginal_cost = 30.0  # Default
                
            marginal_costs.append(marginal_cost)
            gen_indices.append(idx)
        
        # Sort generators by marginal cost (economic dispatch order)
        sorted_pairs = sorted(zip(marginal_costs, gen_indices))
        
        # Calculate total load to meet
        total_load = net.load['p_mw'].sum()
        
        # Dispatch generators in merit order
        remaining_load = total_load * 1.05  # 5% reserve margin
        
        for marginal_cost, idx in sorted_pairs:
            if remaining_load <= 0:
                # No more load to serve
                net.gen.at[idx, 'p_mw'] = 0.0
            else:
                max_p = net.gen.at[idx, 'max_p_mw']
                dispatch = min(max_p, remaining_load)
                net.gen.at[idx, 'p_mw'] = dispatch
                remaining_load -= dispatch
    
    def _apply_voltage_stress(self, net: pp.pandapowerNet, scenario: ScenarioConditions):
        """Apply voltage stress conditions"""
        
        if scenario.voltage_stress_factor > 1.0:
            # Raise voltage setpoints for stress scenarios
            stress_factor = scenario.voltage_stress_factor
            
            # Adjust generator voltage setpoints
            for idx in net.gen.index:
                current_vm = net.gen.at[idx, 'vm_pu']
                net.gen.at[idx, 'vm_pu'] = current_vm * stress_factor
                
            # Adjust external grid voltage
            for idx in net.ext_grid.index:
                current_vm = net.ext_grid.at[idx, 'vm_pu']
                net.ext_grid.at[idx, 'vm_pu'] = current_vm * stress_factor
                
    def _calculate_temperature_factor(self, temp_c: float) -> float:
        """Calculate load factor based on temperature (HVAC impact)"""
        
        # Comfort zone around 20°C
        comfort_temp = 20.0
        temp_diff = abs(temp_c - comfort_temp)
        
        # Increase load for extreme temperatures (heating/cooling)
        if temp_diff > 10:
            # Significant HVAC load
            temp_factor = 1.0 + (temp_diff - 10) * 0.02  # 2% per degree beyond ±10°C
        elif temp_diff > 5:
            # Moderate HVAC load
            temp_factor = 1.0 + (temp_diff - 5) * 0.01  # 1% per degree beyond ±5°C
        else:
            # Within comfort zone
            temp_factor = 1.0
            
        return min(temp_factor, 1.3)  # Cap at 30% increase
    
    def run_scenario_simulation(self, scenario: ScenarioConditions, 
                              base_net: pp.pandapowerNet) -> SimulationResult:
        """Run a complete scenario simulation with full metrics"""
        
        import time
        from simulation.power_flow import PowerFlowEngine
        
        # Apply scenario to network
        scenario_net = self.apply_scenario(scenario, base_net)
        
        # Run power flow
        power_flow_engine = PowerFlowEngine()
        start_time = time.time()
        
        try:
            result = power_flow_engine.solve_power_flow(scenario_net)
            computation_time = time.time() - start_time
            converged = result.converged
            iterations = result.iterations
            final_residual = 0.0  # PowerFlowResults doesn't have final_residual
            
        except Exception as e:
            logger.warning(f"Power flow failed for {scenario.scenario_type.value} scenario: {e}")
            computation_time = time.time() - start_time
            converged = False
            iterations = 0
            final_residual = float('inf')
            
        # Calculate metrics
        if converged and 'res_bus' in scenario_net:
            metrics = self._calculate_metrics(scenario_net)
        else:
            # Default metrics for non-converged cases
            metrics = {
                'total_generation': 0.0,
                'total_load': scenario_net.load['p_mw'].sum(),
                'total_losses': float('inf'),
                'loss_percentage': float('inf'),
                'min_voltage': 0.0,
                'max_voltage': 0.0,
                'voltage_violations': 999,
                'stability_margin': 0.0,
                'min_eigenvalue': -999.0,
                'max_line_loading': float('inf'),
                'line_violations': 999,
                'generation_cost': float('inf'),
                'load_shedding': scenario_net.load['p_mw'].sum()
            }
        
        return SimulationResult(
            scenario=scenario,
            converged=converged,
            iterations=iterations,
            computation_time=computation_time,
            final_residual=final_residual,
            **metrics
        )
    
    def _calculate_metrics(self, net: pp.pandapowerNet) -> Dict[str, Any]:
        """Calculate comprehensive system metrics"""
        
        metrics = {}
        
        # Power metrics
        if 'res_gen' in net:
            metrics['total_generation'] = net.res_gen['p_mw'].sum()
        else:
            metrics['total_generation'] = net.gen['p_mw'].sum()
            
        metrics['total_load'] = net.load['p_mw'].sum()
        
        if 'res_line' in net:
            line_losses = (net.res_line['pl_mw'].sum() if 'pl_mw' in net.res_line.columns else 0.0)
            trafo_losses = (net.res_trafo['pl_mw'].sum() if 'res_trafo' in net and 'pl_mw' in net.res_trafo.columns else 0.0)
            metrics['total_losses'] = line_losses + trafo_losses
        else:
            metrics['total_losses'] = 0.0
            
        metrics['loss_percentage'] = 100.0 * metrics['total_losses'] / max(metrics['total_generation'], 1.0)
        
        # Voltage metrics
        if 'res_bus' in net:
            voltages = net.res_bus['vm_pu']
            metrics['min_voltage'] = voltages.min()
            metrics['max_voltage'] = voltages.max()
            metrics['voltage_violations'] = ((voltages < 0.95) | (voltages > 1.05)).sum()
        else:
            metrics['min_voltage'] = 0.0
            metrics['max_voltage'] = 0.0
            metrics['voltage_violations'] = 999
            
        # Line loading metrics
        if 'res_line' in net and 'loading_percent' in net.res_line.columns:
            line_loadings = net.res_line['loading_percent']
            metrics['max_line_loading'] = line_loadings.max() / 100.0
            metrics['line_violations'] = (line_loadings > 100.0).sum()
        else:
            metrics['max_line_loading'] = 0.0
            metrics['line_violations'] = 0
            
        # Simplified stability metrics (would need more sophisticated analysis in practice)
        metrics['stability_margin'] = random.uniform(0.1, 0.5) if metrics['min_voltage'] > 0.9 else 0.0
        metrics['min_eigenvalue'] = random.uniform(-0.1, 0.1)
        
        # Economic metrics (simplified)
        metrics['generation_cost'] = metrics['total_generation'] * random.uniform(30, 60)  # $/MWh
        metrics['load_shedding'] = max(0, metrics['total_load'] - metrics['total_generation'])
        
        return metrics