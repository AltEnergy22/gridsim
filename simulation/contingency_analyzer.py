"""
Contingency Analysis Engine

Applies contingency scenarios to power system networks and analyzes the results.
Implements RTCA (Real-Time Contingency Analysis) capabilities.
"""

import copy
import time
import pandas as pd
import pandapower as pp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from loguru import logger

from grid.advanced_grid import AdvancedGrid
from grid.grid_components import Contingency
from simulation.power_flow import PowerFlowEngine, PowerFlowResults


@dataclass
class ViolationDetails:
    """Details of a system violation"""
    element_id: str
    element_name: str
    element_type: str  # 'bus', 'line', 'transformer'
    violation_type: str  # 'voltage_high', 'voltage_low', 'thermal_overload'
    severity: float
    limit: float
    actual_value: float
    critical: bool
    location: str  # bus ID or element location


@dataclass  
class ContingencyResult:
    """Results from analyzing a single contingency"""
    contingency: Contingency
    converged: bool
    solve_time: float
    iterations: int
    
    # Pre-contingency state
    pre_violations: List[ViolationDetails]
    
    # Post-contingency state
    post_violations: List[ViolationDetails]
    new_violations: List[ViolationDetails]
    cleared_violations: List[ViolationDetails]
    
    # System metrics
    total_generation_mw: float
    total_load_mw: float
    total_losses_mw: float
    max_voltage_pu: float
    min_voltage_pu: float
    max_line_loading_pct: float
    
    # Islanding information
    islands_created: int
    islanded_buses: List[str]
    
    # Overall assessment
    severity_score: float
    criticality_level: str  # 'low', 'medium', 'high', 'critical'


class ContingencyAnalyzer:
    """
    Comprehensive contingency analysis engine
    
    Applies contingency scenarios to power networks, detects violations,
    and prepares data for mitigation planning.
    """
    
    def __init__(self, grid: AdvancedGrid):
        self.grid = grid
        self.power_flow_engine = PowerFlowEngine()
        
        # Violation thresholds
        self.voltage_limits = {
            'normal_min': 0.95,
            'normal_max': 1.05,
            'emergency_min': 0.90,
            'emergency_max': 1.10
        }
        
        self.thermal_limits = {
            'normal_loading': 100.0,
            'emergency_loading': 120.0,
            'critical_loading': 150.0
        }
        
    def analyze_all_contingencies(self, 
                                base_net: pp.pandapowerNet,
                                contingencies: List[Contingency],
                                parallel: bool = True,
                                max_workers: int = 8) -> List[ContingencyResult]:
        """
        Analyze all contingencies with optional parallelization
        
        Args:
            base_net: Base pandapower network
            contingencies: List of contingency scenarios
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            
        Returns:
            List of contingency analysis results
        """
        
        logger.info(f"Starting analysis of {len(contingencies)} contingencies...")
        
        # Get pre-contingency violations (base case)
        base_result = self.power_flow_engine.solve_power_flow(base_net)
        pre_violations = self._extract_violations(base_net, base_result) if base_result.converged else []
        
        results = []
        
        if parallel and len(contingencies) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for contingency in contingencies:
                    future = executor.submit(
                        self._analyze_single_contingency,
                        contingency,
                        base_net,
                        pre_violations
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per contingency
                        results.append(result)
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"Completed {i + 1}/{len(contingencies)} contingencies")
                            
                    except Exception as e:
                        logger.error(f"Error analyzing contingency {contingencies[i].id}: {e}")
                        # Create failure result
                        result = self._create_failure_result(contingencies[i], pre_violations, str(e))
                        results.append(result)
        else:
            # Sequential processing
            for i, contingency in enumerate(contingencies):
                try:
                    result = self._analyze_single_contingency(contingency, base_net, pre_violations)
                    results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Completed {i + 1}/{len(contingencies)} contingencies")
                        
                except Exception as e:
                    logger.error(f"Error analyzing contingency {contingency.id}: {e}")
                    result = self._create_failure_result(contingency, pre_violations, str(e))
                    results.append(result)
        
        logger.info(f"Completed contingency analysis. {len(results)} results generated.")
        return results
    
    def _analyze_single_contingency(self, 
                                  contingency: Contingency,
                                  base_net: pp.pandapowerNet,
                                  pre_violations: List[ViolationDetails]) -> ContingencyResult:
        """Analyze a single contingency scenario"""
        
        start_time = time.time()
        
        # Create contingency network by removing elements
        cont_net = self._apply_contingency(base_net, contingency)
        
        # Solve power flow
        pf_result = self.power_flow_engine.solve_power_flow(cont_net)
        solve_time = time.time() - start_time
        
        if pf_result.converged:
            # Extract violations and analyze
            post_violations = self._extract_violations(cont_net, pf_result)
            new_violations, cleared_violations = self._compare_violations(pre_violations, post_violations)
            
            # Check for islanding
            islands_created, islanded_buses = self._detect_islanding(cont_net)
            
            # Calculate severity score
            severity_score = self._calculate_severity_score(post_violations, new_violations)
            criticality_level = self._assess_criticality(severity_score, new_violations)
            
            result = ContingencyResult(
                contingency=contingency,
                converged=True,
                solve_time=solve_time,
                iterations=pf_result.iterations,
                pre_violations=pre_violations,
                post_violations=post_violations,
                new_violations=new_violations,
                cleared_violations=cleared_violations,
                total_generation_mw=pf_result.total_generation_mw,
                total_load_mw=pf_result.total_load_mw,
                total_losses_mw=pf_result.total_losses_mw,
                max_voltage_pu=pf_result.max_voltage_pu,
                min_voltage_pu=pf_result.min_voltage_pu,
                max_line_loading_pct=pf_result.max_line_loading_pct,
                islands_created=islands_created,
                islanded_buses=islanded_buses,
                severity_score=severity_score,
                criticality_level=criticality_level
            )
        else:
            # Power flow did not converge
            result = self._create_failure_result(contingency, pre_violations, "Power flow diverged")
            result.solve_time = solve_time
            result.iterations = pf_result.iterations
        
        return result
    
    def _apply_contingency(self, base_net: pp.pandapowerNet, contingency: Contingency) -> pp.pandapowerNet:
        """Apply contingency by removing specified elements from network"""
        
        # Deep copy the network
        cont_net = copy.deepcopy(base_net)
        
        for element_id in contingency.elements:
            try:
                # Determine element type and remove appropriately
                if element_id.startswith('L_') or element_id.startswith('LINE_'):
                    # Line outage
                    self._remove_line(cont_net, element_id)
                elif element_id.startswith('GEN_') or element_id.startswith('G_'):
                    # Generator trip
                    self._remove_generator(cont_net, element_id)
                elif element_id.startswith('XFMR_') or element_id.startswith('T_'):
                    # Transformer failure
                    self._remove_transformer(cont_net, element_id)
                else:
                    logger.warning(f"Unknown element type for {element_id}")
                    
            except Exception as e:
                logger.error(f"Error removing element {element_id}: {e}")
        
        return cont_net
    
    def _remove_line(self, net: pp.pandapowerNet, line_id: str):
        """Remove a line from the network"""
        # Find line by name
        line_mask = net.line['name'] == line_id
        if line_mask.any():
            line_indices = net.line.index[line_mask].tolist()
            for idx in line_indices:
                net.line.at[idx, 'in_service'] = False
        else:
            logger.warning(f"Line {line_id} not found in network")
    
    def _remove_generator(self, net: pp.pandapowerNet, gen_id: str):
        """Remove a generator from the network"""
        # Find generator by name
        gen_mask = net.gen['name'] == gen_id
        if gen_mask.any():
            gen_indices = net.gen.index[gen_mask].tolist()
            for idx in gen_indices:
                net.gen.at[idx, 'in_service'] = False
        else:
            logger.warning(f"Generator {gen_id} not found in network")
    
    def _remove_transformer(self, net: pp.pandapowerNet, xfmr_id: str):
        """Remove a transformer from the network"""
        # Find transformer by name
        trafo_mask = net.trafo['name'] == xfmr_id
        if trafo_mask.any():
            trafo_indices = net.trafo.index[trafo_mask].tolist()
            for idx in trafo_indices:
                net.trafo.at[idx, 'in_service'] = False
        else:
            logger.warning(f"Transformer {xfmr_id} not found in network")
    
    def _extract_violations(self, net: pp.pandapowerNet, pf_result: PowerFlowResults) -> List[ViolationDetails]:
        """Extract all violations from power flow results"""
        violations = []
        
        # Voltage violations
        for idx, row in net.res_bus.iterrows():
            voltage_pu = row['vm_pu']
            bus_name = net.bus.loc[idx, 'name']
            
            if voltage_pu < self.voltage_limits['normal_min']:
                violations.append(ViolationDetails(
                    element_id=bus_name,
                    element_name=bus_name,
                    element_type='bus',
                    violation_type='voltage_low',
                    severity=self.voltage_limits['normal_min'] - voltage_pu,
                    limit=self.voltage_limits['normal_min'],
                    actual_value=voltage_pu,
                    critical=voltage_pu < self.voltage_limits['emergency_min'],
                    location=bus_name
                ))
            elif voltage_pu > self.voltage_limits['normal_max']:
                violations.append(ViolationDetails(
                    element_id=bus_name,
                    element_name=bus_name,
                    element_type='bus',
                    violation_type='voltage_high',
                    severity=voltage_pu - self.voltage_limits['normal_max'],
                    limit=self.voltage_limits['normal_max'],
                    actual_value=voltage_pu,
                    critical=voltage_pu > self.voltage_limits['emergency_max'],
                    location=bus_name
                ))
        
        # Line thermal violations
        if hasattr(net, 'res_line') and len(net.res_line) > 0:
            for idx, row in net.res_line.iterrows():
                loading_pct = row['loading_percent']
                line_name = net.line.loc[idx, 'name']
                
                if loading_pct > self.thermal_limits['normal_loading']:
                    violations.append(ViolationDetails(
                        element_id=line_name,
                        element_name=line_name,
                        element_type='line',
                        violation_type='thermal_overload',
                        severity=loading_pct - self.thermal_limits['normal_loading'],
                        limit=self.thermal_limits['normal_loading'],
                        actual_value=loading_pct,
                        critical=loading_pct > self.thermal_limits['critical_loading'],
                        location=f"{net.line.loc[idx, 'from_bus']}-{net.line.loc[idx, 'to_bus']}"
                    ))
        
        # Transformer thermal violations  
        if hasattr(net, 'res_trafo') and len(net.res_trafo) > 0:
            for idx, row in net.res_trafo.iterrows():
                loading_pct = row['loading_percent']
                trafo_name = net.trafo.loc[idx, 'name']
                
                if loading_pct > self.thermal_limits['normal_loading']:
                    violations.append(ViolationDetails(
                        element_id=trafo_name,
                        element_name=trafo_name,
                        element_type='transformer',
                        violation_type='thermal_overload',
                        severity=loading_pct - self.thermal_limits['normal_loading'],
                        limit=self.thermal_limits['normal_loading'],
                        actual_value=loading_pct,
                        critical=loading_pct > self.thermal_limits['critical_loading'],
                        location=f"{net.trafo.loc[idx, 'hv_bus']}-{net.trafo.loc[idx, 'lv_bus']}"
                    ))
        
        return violations
    
    def _compare_violations(self, pre_violations: List[ViolationDetails], 
                          post_violations: List[ViolationDetails]) -> Tuple[List[ViolationDetails], List[ViolationDetails]]:
        """Compare pre and post contingency violations"""
        
        # Create sets for comparison
        pre_violation_keys = {(v.element_id, v.violation_type) for v in pre_violations}
        post_violation_keys = {(v.element_id, v.violation_type) for v in post_violations}
        
        # New violations
        new_violation_keys = post_violation_keys - pre_violation_keys
        new_violations = [v for v in post_violations 
                         if (v.element_id, v.violation_type) in new_violation_keys]
        
        # Cleared violations  
        cleared_violation_keys = pre_violation_keys - post_violation_keys
        cleared_violations = [v for v in pre_violations
                            if (v.element_id, v.violation_type) in cleared_violation_keys]
        
        return new_violations, cleared_violations
    
    def _detect_islanding(self, net: pp.pandapowerNet) -> Tuple[int, List[str]]:
        """Detect network islanding after contingency"""
        
        try:
            # Use pandapower's connectivity check
            if hasattr(pp.topology, 'connected_components'):
                components = pp.topology.connected_components(net)
                islands_created = len(components) - 1  # Subtract 1 for main island
                
                # Get buses in smaller islands
                islanded_buses = []
                if len(components) > 1:
                    # Find largest component (main grid)
                    main_component = max(components, key=len)
                    
                    # All other buses are islanded
                    for component in components:
                        if component != main_component:
                            islanded_buses.extend([net.bus.loc[idx, 'name'] for idx in component])
                
                return islands_created, islanded_buses
            else:
                # Fallback: simple connectivity check
                return 0, []
                
        except Exception as e:
            logger.warning(f"Error detecting islanding: {e}")
            return 0, []
    
    def _calculate_severity_score(self, post_violations: List[ViolationDetails], 
                                new_violations: List[ViolationDetails]) -> float:
        """Calculate overall severity score for the contingency"""
        
        score = 0.0
        
        # Weight different violation types
        violation_weights = {
            'voltage_low': 2.0,
            'voltage_high': 1.5,
            'thermal_overload': 1.0
        }
        
        # Score based on new violations (worse than existing ones)
        for violation in new_violations:
            weight = violation_weights.get(violation.violation_type, 1.0)
            severity_factor = violation.severity
            critical_factor = 2.0 if violation.critical else 1.0
            
            score += weight * severity_factor * critical_factor
        
        # Additional penalty for total violation count
        score += len(post_violations) * 0.1
        
        return score
    
    def _assess_criticality(self, severity_score: float, new_violations: List[ViolationDetails]) -> str:
        """Assess criticality level based on severity and violation characteristics"""
        
        # Check for critical violations
        has_critical = any(v.critical for v in new_violations)
        
        if has_critical or severity_score > 10.0:
            return 'critical'
        elif severity_score > 5.0 or len(new_violations) > 10:
            return 'high'
        elif severity_score > 1.0 or len(new_violations) > 3:
            return 'medium'
        else:
            return 'low'
    
    def _create_failure_result(self, contingency: Contingency, 
                             pre_violations: List[ViolationDetails],
                             error_msg: str) -> ContingencyResult:
        """Create a result object for failed contingency analysis"""
        
        return ContingencyResult(
            contingency=contingency,
            converged=False,
            solve_time=0.0,
            iterations=0,
            pre_violations=pre_violations,
            post_violations=[],
            new_violations=[],
            cleared_violations=[],
            total_generation_mw=0.0,
            total_load_mw=0.0,
            total_losses_mw=0.0,
            max_voltage_pu=0.0,
            min_voltage_pu=0.0,
            max_line_loading_pct=0.0,
            islands_created=0,
            islanded_buses=[],
            severity_score=999.0,  # High score indicates failure
            criticality_level='critical'
        )
    
    def export_results_to_dataframe(self, results: List[ContingencyResult]) -> pd.DataFrame:
        """Export contingency results to pandas DataFrame for analysis"""
        
        records = []
        
        for result in results:
            # Basic contingency info
            record = {
                'contingency_id': result.contingency.id,
                'contingency_type': result.contingency.type,
                'elements': ','.join(result.contingency.elements),
                'probability': result.contingency.probability,
                'converged': result.converged,
                'solve_time': result.solve_time,
                'iterations': result.iterations,
                
                # System metrics
                'total_generation_mw': result.total_generation_mw,
                'total_load_mw': result.total_load_mw,
                'total_losses_mw': result.total_losses_mw,
                'max_voltage_pu': result.max_voltage_pu,
                'min_voltage_pu': result.min_voltage_pu,
                'max_line_loading_pct': result.max_line_loading_pct,
                
                # Violation counts
                'pre_violations': len(result.pre_violations),
                'post_violations': len(result.post_violations),
                'new_violations': len(result.new_violations),
                'cleared_violations': len(result.cleared_violations),
                
                # Critical violation counts
                'critical_violations': len([v for v in result.post_violations if v.critical]),
                'new_critical_violations': len([v for v in result.new_violations if v.critical]),
                
                # Islanding
                'islands_created': result.islands_created,
                'islanded_buses_count': len(result.islanded_buses),
                
                # Assessment
                'severity_score': result.severity_score,
                'criticality_level': result.criticality_level
            }
            
            records.append(record)
        
        return pd.DataFrame(records)


def screen_critical_contingencies(results: List[ContingencyResult], 
                                top_n: int = 50) -> List[ContingencyResult]:
    """Screen and return the most critical contingencies"""
    
    # Filter to converged results with violations
    critical_results = [r for r in results if r.converged and len(r.new_violations) > 0]
    
    # Sort by severity score
    critical_results.sort(key=lambda x: x.severity_score, reverse=True)
    
    return critical_results[:top_n]


if __name__ == "__main__":
    # Example usage
    from grid.advanced_grid import AdvancedGrid
    from simulation.contingency_generator import ContingencyGenerator, load_contingencies
    from simulation.build_network import build_pandapower_network
    
    # Create test grid
    regions = ['A', 'B', 'C']
    grid = AdvancedGrid(regions, buses_per_region=100)
    
    # Build pandapower network
    base_net, _ = build_pandapower_network(grid)
    
    # Load contingencies (assuming they were generated)
    contingencies = load_contingencies("data/contingencies")[:10]  # Test with first 10
    
    if contingencies:
        # Analyze contingencies
        analyzer = ContingencyAnalyzer(grid)
        results = analyzer.analyze_all_contingencies(base_net, contingencies)
        
        # Export results
        df = analyzer.export_results_to_dataframe(results)
        print(f"Analyzed {len(results)} contingencies")
        print(f"Converged: {df['converged'].sum()}")
        print(f"Critical contingencies: {len([r for r in results if r.criticality_level == 'critical'])}")
    else:
        print("No contingencies found. Run contingency generation first.") 