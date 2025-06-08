"""
Realistic Grid Simulation Orchestrator

Orchestrates the complete pipeline:
1. Generate diverse operational scenarios
2. Apply scenarios to power system network
3. Run simulations with convergence tracking
4. Label and export results for ML training
"""

import pandas as pd
import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
import json
from pathlib import Path

from loguru import logger
import pandapower as pp

from grid.advanced_grid import AdvancedGrid
from simulation.build_network import build_pandapower_network
from simulation.scenario_generator import ScenarioGenerator, ScenarioType, ScenarioConditions
from simulation.scenario_application import ScenarioApplication, SimulationResult

class RealisticSimulationOrchestrator:
    """Orchestrate realistic power system simulation campaigns"""
    
    def __init__(self, grid_size: int = 2000):
        self.grid_size = grid_size
        self.grid = None
        self.base_net = None
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator()
        self.scenario_application = ScenarioApplication()
        
        # Results storage
        self.simulation_results: List[SimulationResult] = []
        
    def setup_base_grid(self):
        """Set up the base grid and network"""
        
        logger.info(f"Setting up base {self.grid_size}-bus grid...")
        
        # Create advanced grid with appropriate regions
        num_regions = max(1, self.grid_size // 400)  # ~400 buses per region
        regions = [f"REGION_{i+1}" for i in range(num_regions)]
        buses_per_region = self.grid_size // num_regions
        
        self.grid = AdvancedGrid(regions=regions, buses_per_region=buses_per_region)
        
        # Build pandapower network
        start_time = time.time()
        import datetime
        timestamp = datetime.datetime.now()
        self.base_net, _ = build_pandapower_network(self.grid, timestamp)
        build_time = time.time() - start_time
        
        logger.info(f"Base network built in {build_time:.2f}s:")
        logger.info(f"  - {len(self.base_net.bus)} buses")
        logger.info(f"  - {len(self.base_net.line)} lines") 
        logger.info(f"  - {len(self.base_net.gen)} generators")
        logger.info(f"  - {len(self.base_net.load)} loads")
        
    def run_simulation_campaign(self, num_scenarios: int = 100) -> pd.DataFrame:
        """Run a campaign of diverse scenarios for ML training"""
        
        if self.base_net is None:
            self.setup_base_grid()
            
        logger.info(f"Starting simulation campaign with {num_scenarios} scenarios...")
        start_time = time.time()
        
        # Generate scenarios
        scenarios = self.scenario_generator.generate_scenario_batch(num_scenarios)
        
        # Run simulations
        results = []
        failed_count = 0
        
        for i, scenario in enumerate(scenarios):
            try:
                # Run scenario simulation
                result = self.scenario_application.run_scenario_simulation(
                    scenario, self.base_net
                )
                
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    conv_rate = (len(results) - failed_count) / len(results) * 100
                    logger.info(f"Progress: {i+1}/{num_scenarios} scenarios "
                              f"({conv_rate:.1f}% convergence)")
                    
            except Exception as e:
                logger.error(f"Scenario {i+1} failed: {e}")
                failed_count += 1
                continue
                
        total_time = time.time() - start_time
        
        # Log campaign summary
        converged_count = sum(1 for r in results if r.converged)
        convergence_rate = converged_count / len(results) * 100 if results else 0
        
        logger.info(f"Campaign completed in {total_time:.2f}s:")
        logger.info(f"  - {len(results)} successful simulations")
        logger.info(f"  - {converged_count} converged ({convergence_rate:.1f}%)")
        logger.info(f"  - {failed_count} failed simulations")
        
        # Convert to DataFrame for analysis
        df = self.results_to_dataframe(results)
        
        # Store results
        self.simulation_results.extend(results)
        
        return df
        
    def results_to_dataframe(self, results: List[SimulationResult]) -> pd.DataFrame:
        """Convert simulation results to a structured DataFrame"""
        
        rows = []
        
        for result in results:
            # Flatten scenario conditions
            scenario_dict = asdict(result.scenario)
            
            # Create row with all metrics
            row = {
                # Scenario metadata
                'timestamp': scenario_dict['timestamp'],
                'scenario_type': scenario_dict['scenario_type'].value,
                'load_multiplier': scenario_dict['load_multiplier'],
                'weather_temp_c': scenario_dict['weather_temp_c'],
                'economic_index': scenario_dict['economic_index'],
                'voltage_stress_factor': scenario_dict['voltage_stress_factor'],
                'expected_convergence': scenario_dict['expected_convergence'],
                'stress_level': scenario_dict['stress_level'],
                
                # Renewable capacity factors
                'solar_cf': scenario_dict['renewable_cf'].get('solar', 0.0),
                'wind_cf': scenario_dict['renewable_cf'].get('wind', 0.0),
                'hydro_cf': scenario_dict['renewable_cf'].get('hydro', 0.0),
                
                # Fuel costs
                'gas_cost': scenario_dict['fuel_costs'].get('gas', 0.0),
                'coal_cost': scenario_dict['fuel_costs'].get('coal', 0.0),
                'nuclear_cost': scenario_dict['fuel_costs'].get('nuclear', 0.0),
                'oil_cost': scenario_dict['fuel_costs'].get('oil', 0.0),
                
                # Simulation results
                'se_converged': result.converged,  # State estimation convergence flag
                'se_iterations': result.iterations,
                'computation_time': result.computation_time,
                'final_residual': result.final_residual,
                
                # Power system metrics
                'total_generation': result.total_generation,
                'total_load': result.total_load,
                'total_losses': result.total_losses,
                'loss_percentage': result.loss_percentage,
                
                # Voltage metrics
                'min_voltage': result.min_voltage,
                'max_voltage': result.max_voltage,
                'voltage_violations': result.voltage_violations,
                
                # Line loading metrics  
                'max_line_loading': result.max_line_loading,
                'line_violations': result.line_violations,
                
                # Stability metrics (simplified)
                'stability_margin': result.stability_margin,
                'min_eigenvalue': result.min_eigenvalue,
                
                # Economic metrics
                'generation_cost': result.generation_cost,
                'load_shedding': result.load_shedding
            }
            
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def export_training_dataset(self, df: pd.DataFrame, filename: str = "simulation_dataset.csv"):
        """Export labeled dataset for ML training"""
        
        # Add additional ML features
        df = self._add_ml_features(df)
        
        # Export to CSV
        output_path = Path("outputs") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Training dataset exported to {output_path}")
        
        # Export summary statistics
        self._export_dataset_summary(df, output_path.with_suffix('.json'))
        
        return output_path
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features useful for ML training"""
        
        df = df.copy()
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Load profile categorization
        df['load_category'] = pd.cut(df['load_multiplier'], 
                                   bins=[0, 0.6, 0.9, 1.1, 1.3, 100],
                                   labels=['very_low', 'low', 'normal', 'high', 'very_high'])
        
        # Renewable penetration
        total_renewable_cf = df['solar_cf'] + df['wind_cf'] + df['hydro_cf']
        df['renewable_penetration'] = total_renewable_cf / 3.0  # Normalized
        
        # System stress indicators
        df['voltage_stress'] = (df['min_voltage'] < 0.95) | (df['max_voltage'] > 1.05)
        df['thermal_stress'] = df['max_line_loading'] > 0.9
        df['overall_stress'] = df['voltage_stress'] | df['thermal_stress']
        
        # Economic stress
        df['fuel_cost_index'] = (df['gas_cost'] + df['coal_cost'] + df['oil_cost']) / 3.0
        df['economic_stress'] = df['economic_index'] > 1.2
        
        # Convergence difficulty prediction features
        df['load_generation_ratio'] = df['total_load'] / (df['total_generation'] + 1e-6)
        df['loss_ratio'] = df['total_losses'] / (df['total_generation'] + 1e-6)
        
        return df
    
    def _export_dataset_summary(self, df: pd.DataFrame, summary_path: Path):
        """Export dataset summary statistics"""
        
        summary = {
            'dataset_info': {
                'total_scenarios': len(df),
                'convergence_rate': (df['se_converged'].sum() / len(df) * 100) if len(df) > 0 else 0,
                'scenario_type_distribution': df['scenario_type'].value_counts().to_dict(),
                'stress_level_distribution': df['stress_level'].value_counts().to_dict(),
            },
            'convergence_analysis': {
                'normal_convergence_rate': (df[df['scenario_type'] == 'normal']['se_converged'].mean() * 100) if len(df[df['scenario_type'] == 'normal']) > 0 else 0,
                'stress_convergence_rate': (df[df['scenario_type'] == 'stress']['se_converged'].mean() * 100) if len(df[df['scenario_type'] == 'stress']) > 0 else 0,
                'challenge_convergence_rate': (df[df['scenario_type'] == 'challenge']['se_converged'].mean() * 100) if len(df[df['scenario_type'] == 'challenge']) > 0 else 0,
            },
            'system_metrics': {
                'avg_computation_time': df['computation_time'].mean(),
                'avg_iterations': df['se_iterations'].mean(),
                'avg_loss_percentage': df[df['se_converged']]['loss_percentage'].mean() if df['se_converged'].any() else None,
                'voltage_violation_rate': (df['voltage_violations'] > 0).mean() * 100,
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Dataset summary exported to {summary_path}")
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze simulation results for insights"""
        
        analysis = {}
        
        # Convergence analysis by scenario type
        conv_by_type = df.groupby('scenario_type')['se_converged'].agg(['count', 'sum', 'mean']).round(3)
        analysis['convergence_by_scenario_type'] = conv_by_type.to_dict('index')
        
        # Performance metrics
        converged_df = df[df['se_converged']]
        if len(converged_df) > 0:
            analysis['performance_metrics'] = {
                'avg_computation_time': converged_df['computation_time'].mean(),
                'avg_iterations': converged_df['se_iterations'].mean(),
                'avg_loss_percentage': converged_df['loss_percentage'].mean(),
                'avg_voltage_violations': converged_df['voltage_violations'].mean(),
            }
        
        # Correlation analysis (only if we have numeric data and the convergence column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'se_converged' in df.columns and len(numeric_cols) > 1:
            try:
                correlation_matrix = df[numeric_cols].corr()['se_converged'].sort_values(ascending=False)
                analysis['convergence_correlations'] = correlation_matrix.to_dict()
            except KeyError:
                analysis['convergence_correlations'] = "Could not compute correlations - insufficient numeric data"
        else:
            analysis['convergence_correlations'] = "Convergence column not found or insufficient data"
        
        return analysis

def main():
    """Example usage of the realistic simulation orchestrator"""
    
    # Configure logging
    logger.add("simulation_campaign.log", rotation="100 MB")
    
    # Create orchestrator
    orchestrator = RealisticSimulationOrchestrator(grid_size=2000)
    
    # Run simulation campaign
    logger.info("Starting realistic simulation campaign...")
    
    # Small test run first
    df = orchestrator.run_simulation_campaign(num_scenarios=20)
    
    # Export training dataset
    output_path = orchestrator.export_training_dataset(df)
    
    # Analyze results
    analysis = orchestrator.analyze_results(df)
    logger.info("Analysis results:")
    for key, value in analysis.items():
        logger.info(f"{key}: {value}")
    
    logger.info(f"Training dataset available at: {output_path}")

if __name__ == "__main__":
    main()