"""
Power Flow Engine - Advanced Power Flow Analysis

This module provides comprehensive power flow analysis capabilities
including multiple algorithms, convergence monitoring, and violation detection.
"""

import pandapower as pp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import time
from dataclasses import dataclass
from enum import Enum

class PowerFlowAlgorithm(Enum):
    """Power flow solution algorithms"""
    NEWTON_RAPHSON = "nr"
    FAST_DECOUPLED = "fdpf"
    GAUSS_SEIDEL = "gs"
    IWAMOTO_NEWTON = "iwamoto_nr"

class ConvergenceStatus(Enum):
    """Power flow convergence status"""
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged" 
    NUMERICAL_ERROR = "numerical_error"

@dataclass
class PowerFlowResults:
    """Comprehensive power flow results"""
    converged: bool
    status: ConvergenceStatus
    iterations: int
    solve_time_sec: float
    algorithm_used: PowerFlowAlgorithm
    
    # Voltage results
    bus_voltages_pu: pd.DataFrame
    voltage_violations: List[Dict[str, Any]]
    
    # Loading results
    line_loadings: pd.DataFrame
    transformer_loadings: pd.DataFrame
    thermal_violations: List[Dict[str, Any]]
    
    # Generation results
    generator_dispatch: pd.DataFrame
    total_generation_mw: float
    total_load_mw: float
    total_losses_mw: float
    
    # System metrics
    max_voltage_pu: float
    min_voltage_pu: float
    max_line_loading_pct: float
    max_trafo_loading_pct: float

class PowerFlowEngine:
    """
    Advanced power flow analysis engine
    """
    
    def __init__(self):
        self.default_algorithms = [
            PowerFlowAlgorithm.NEWTON_RAPHSON,
            PowerFlowAlgorithm.IWAMOTO_NEWTON
        ]
        
        self.voltage_limits = {
            'min_pu': 0.95,
            'max_pu': 1.05,
            'critical_min_pu': 0.90,
            'critical_max_pu': 1.10
        }
        
        self.thermal_limits = {
            'line_emergency_pct': 120,
            'trafo_emergency_pct': 130
        }
    
    def solve_power_flow(self, net: pp.pandapowerNet,
                        algorithm: Optional[PowerFlowAlgorithm] = None,
                        max_iteration: int = 50,
                        tolerance_mva: float = 1e-5,
                        enforce_q_lims: bool = True) -> PowerFlowResults:
        """
        Solve power flow with advanced convergence strategies
        
        Args:
            net: Pandapower network
            algorithm: Power flow algorithm to use (if None, tries multiple)
            max_iteration: Maximum iterations
            tolerance_mva: Convergence tolerance in MVA
            enforce_q_lims: Whether to enforce generator reactive power limits
            
        Returns:
            Comprehensive power flow results
        """
        logger.info("Starting power flow analysis...")
        
        start_time = time.time()
        
        # Try algorithms in order until one converges
        algorithms_to_try = [algorithm] if algorithm else self.default_algorithms
        
        result = None
        for alg in algorithms_to_try:
            logger.info(f"Trying algorithm: {alg.value}")
            
            try:
                # Run power flow
                pp.runpp(net, algorithm=alg.value, max_iteration=max_iteration, 
                        tolerance_mva=tolerance_mva, enforce_q_lims=enforce_q_lims)
                
                # Check convergence
                if net.converged:
                    solve_time = time.time() - start_time
                    logger.info(f"Power flow converged with {alg.value} in {solve_time:.3f}s")
                    
                    result = self._analyze_results(net, alg, max_iteration, solve_time)
                    break
                else:
                    logger.warning(f"Power flow did not converge with {alg.value}")
                    
            except Exception as e:
                logger.error(f"Power flow failed with {alg.value}: {str(e)}")
                continue
        
        # If no algorithm worked, return failure result
        if result is None:
            solve_time = time.time() - start_time
            result = self._create_failure_result(algorithms_to_try[-1], solve_time)
        
        return result
    
    def _analyze_results(self, net: pp.pandapowerNet, 
                        algorithm: PowerFlowAlgorithm,
                        iterations: int,
                        solve_time: float) -> PowerFlowResults:
        """Analyze power flow results and detect violations"""
        
        # Bus voltage analysis
        bus_voltages = net.res_bus[['vm_pu', 'va_degree']].copy()
        bus_voltages['bus_name'] = net.bus['name']
        
        voltage_violations = self._detect_voltage_violations(net)
        
        # Line loading analysis
        line_loadings = net.res_line[['loading_percent', 'p_from_mw', 'q_from_mvar']].copy()
        line_loadings['line_name'] = net.line['name']
        
        # Transformer loading analysis
        trafo_loadings = net.res_trafo[['loading_percent', 'p_hv_mw', 'q_hv_mvar']].copy()
        trafo_loadings['trafo_name'] = net.trafo['name']
        
        thermal_violations = self._detect_thermal_violations(net)
        
        # Generator dispatch analysis
        generator_dispatch = net.res_gen[['p_mw', 'q_mvar', 'vm_pu']].copy()
        generator_dispatch['gen_name'] = net.gen['name']
        
        # System totals
        total_gen_mw = net.res_gen['p_mw'].sum()
        total_load_mw = net.res_load['p_mw'].sum()
        total_losses_mw = self._calculate_total_losses(net)
        
        # System metrics
        max_voltage = net.res_bus['vm_pu'].max()
        min_voltage = net.res_bus['vm_pu'].min()
        max_line_loading = net.res_line['loading_percent'].max() if len(net.res_line) > 0 else 0.0
        max_trafo_loading = net.res_trafo['loading_percent'].max() if len(net.res_trafo) > 0 else 0.0
        
        return PowerFlowResults(
            converged=True,
            status=ConvergenceStatus.CONVERGED,
            iterations=iterations,
            solve_time_sec=solve_time,
            algorithm_used=algorithm,
            bus_voltages_pu=bus_voltages,
            voltage_violations=voltage_violations,
            line_loadings=line_loadings,
            transformer_loadings=trafo_loadings,
            thermal_violations=thermal_violations,
            generator_dispatch=generator_dispatch,
            total_generation_mw=total_gen_mw,
            total_load_mw=total_load_mw,
            total_losses_mw=total_losses_mw,
            max_voltage_pu=max_voltage,
            min_voltage_pu=min_voltage,
            max_line_loading_pct=max_line_loading,
            max_trafo_loading_pct=max_trafo_loading
        )
    
    def _detect_voltage_violations(self, net: pp.pandapowerNet) -> List[Dict[str, Any]]:
        """Detect voltage limit violations"""
        violations = []
        
        for idx, row in net.res_bus.iterrows():
            voltage_pu = row['vm_pu']
            bus_name = net.bus.loc[idx, 'name']
            
            if voltage_pu < self.voltage_limits['min_pu']:
                severity = "critical" if voltage_pu < self.voltage_limits['critical_min_pu'] else "normal"
                violations.append({
                    'bus_idx': idx,
                    'bus_name': bus_name,
                    'violation_type': 'undervoltage',
                    'voltage_pu': voltage_pu,
                    'limit_pu': self.voltage_limits['min_pu'],
                    'deviation_pu': self.voltage_limits['min_pu'] - voltage_pu,
                    'severity': severity
                })
            
            elif voltage_pu > self.voltage_limits['max_pu']:
                severity = "critical" if voltage_pu > self.voltage_limits['critical_max_pu'] else "normal"
                violations.append({
                    'bus_idx': idx,
                    'bus_name': bus_name,
                    'violation_type': 'overvoltage',
                    'voltage_pu': voltage_pu,
                    'limit_pu': self.voltage_limits['max_pu'],
                    'deviation_pu': voltage_pu - self.voltage_limits['max_pu'],
                    'severity': severity
                })
        
        return violations
    
    def _detect_thermal_violations(self, net: pp.pandapowerNet) -> List[Dict[str, Any]]:
        """Detect thermal limit violations"""
        violations = []
        
        # Line violations
        for idx, row in net.res_line.iterrows():
            loading_pct = row['loading_percent']
            line_name = net.line.loc[idx, 'name']
            
            if loading_pct > 100.0:
                severity = "critical" if loading_pct > self.thermal_limits['line_emergency_pct'] else "normal"
                violations.append({
                    'element_type': 'line',
                    'element_idx': idx,
                    'element_name': line_name,
                    'violation_type': 'thermal_overload',
                    'loading_percent': loading_pct,
                    'overload_percent': loading_pct - 100.0,
                    'severity': severity,
                    'p_flow_mw': row['p_from_mw'],
                    'q_flow_mvar': row['q_from_mvar']
                })
        
        # Transformer violations
        for idx, row in net.res_trafo.iterrows():
            loading_pct = row['loading_percent']
            trafo_name = net.trafo.loc[idx, 'name']
            
            if loading_pct > 100.0:
                severity = "critical" if loading_pct > self.thermal_limits['trafo_emergency_pct'] else "normal"
                violations.append({
                    'element_type': 'transformer',
                    'element_idx': idx,
                    'element_name': trafo_name,
                    'violation_type': 'thermal_overload',
                    'loading_percent': loading_pct,
                    'overload_percent': loading_pct - 100.0,
                    'severity': severity,
                    'p_flow_mw': row['p_hv_mw'],
                    'q_flow_mvar': row['q_hv_mvar']
                })
        
        return violations
    
    def _calculate_total_losses(self, net: pp.pandapowerNet) -> float:
        """Calculate total system losses"""
        line_losses = net.res_line['pl_mw'].sum() if len(net.res_line) > 0 else 0.0
        trafo_losses = net.res_trafo['pl_mw'].sum() if len(net.res_trafo) > 0 else 0.0
        return line_losses + trafo_losses
    
    def _create_failure_result(self, algorithm: PowerFlowAlgorithm, 
                              solve_time: float) -> PowerFlowResults:
        """Create result object for failed power flow"""
        return PowerFlowResults(
            converged=False,
            status=ConvergenceStatus.DIVERGED,
            iterations=-1,
            solve_time_sec=solve_time,
            algorithm_used=algorithm,
            bus_voltages_pu=pd.DataFrame(),
            voltage_violations=[],
            line_loadings=pd.DataFrame(),
            transformer_loadings=pd.DataFrame(),
            thermal_violations=[],
            generator_dispatch=pd.DataFrame(),
            total_generation_mw=0.0,
            total_load_mw=0.0,
            total_losses_mw=0.0,
            max_voltage_pu=0.0,
            min_voltage_pu=0.0,
            max_line_loading_pct=0.0,
            max_trafo_loading_pct=0.0
        )
    
    def run_multiple_scenarios(self, base_net: pp.pandapowerNet,
                              scenarios: List[Dict[str, Any]]) -> Dict[str, PowerFlowResults]:
        """
        Run power flow for multiple scenarios
        
        Args:
            base_net: Base pandapower network
            scenarios: List of scenario definitions
            
        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', f"scenario_{len(results)}")
            logger.info(f"Running scenario: {scenario_name}")
            
            # Copy network for this scenario
            net = base_net.deepcopy()
            
            # Apply scenario modifications
            self._apply_scenario(net, scenario)
            
            # Solve power flow
            result = self.solve_power_flow(net)
            results[scenario_name] = result
        
        return results
    
    def _apply_scenario(self, net: pp.pandapowerNet, scenario: Dict[str, Any]):
        """Apply scenario modifications to network"""
        modifications = scenario.get('modifications', [])
        
        for mod in modifications:
            element_type = mod['element_type']
            element_idx = mod['element_idx']
            parameter = mod['parameter']
            value = mod['value']
            
            if element_type == 'load':
                net.load.loc[element_idx, parameter] = value
            elif element_type == 'gen':
                net.gen.loc[element_idx, parameter] = value
            elif element_type == 'line':
                net.line.loc[element_idx, parameter] = value
            elif element_type == 'trafo':
                net.trafo.loc[element_idx, parameter] = value
    
    def export_results(self, results: PowerFlowResults, 
                      output_path: str, format: str = 'excel'):
        """
        Export power flow results to file
        
        Args:
            results: Power flow results
            output_path: Output file path
            format: Export format ('excel', 'csv')
        """
        if format == 'excel':
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                results.bus_voltages_pu.to_excel(writer, sheet_name='Bus_Voltages')
                results.line_loadings.to_excel(writer, sheet_name='Line_Loadings')
                results.transformer_loadings.to_excel(writer, sheet_name='Transformer_Loadings')
                results.generator_dispatch.to_excel(writer, sheet_name='Generator_Dispatch')
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Converged', 'Algorithm', 'Iterations', 'Solve Time (s)',
                        'Total Generation (MW)', 'Total Load (MW)', 'Total Losses (MW)',
                        'Max Voltage (pu)', 'Min Voltage (pu)',
                        'Max Line Loading (%)', 'Max Trafo Loading (%)',
                        'Voltage Violations', 'Thermal Violations'
                    ],
                    'Value': [
                        results.converged, results.algorithm_used.value, results.iterations,
                        results.solve_time_sec, results.total_generation_mw, results.total_load_mw,
                        results.total_losses_mw, results.max_voltage_pu, results.min_voltage_pu,
                        results.max_line_loading_pct, results.max_trafo_loading_pct,
                        len(results.voltage_violations), len(results.thermal_violations)
                    ]
                }
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Violations sheets
                if results.voltage_violations:
                    pd.DataFrame(results.voltage_violations).to_excel(
                        writer, sheet_name='Voltage_Violations', index=False)
                
                if results.thermal_violations:
                    pd.DataFrame(results.thermal_violations).to_excel(
                        writer, sheet_name='Thermal_Violations', index=False)
        
        logger.info(f"Power flow results exported to {output_path}")

# Convenience function for simple power flow
def run_power_flow(net: pp.pandapowerNet, **kwargs) -> PowerFlowResults:
    """
    Convenience function to run power flow analysis
    
    Args:
        net: pandapower network
        **kwargs: Additional options for power flow engine
        
    Returns:
        Power flow results
    """
    engine = PowerFlowEngine()
    return engine.solve_power_flow(net, **kwargs) 