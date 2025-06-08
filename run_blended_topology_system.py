"""
Blended Topology Grid Simulation System - Main Orchestrator

Complete end-to-end implementation of the comprehensive grid simulation system
featuring contingency analysis, mitigation planning, and ML integration.

Usage:
    python run_blended_topology_system.py --mode full
    python run_blended_topology_system.py --mode contingency_only
    python run_blended_topology_system.py --mode ml_only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Core grid components
from grid.advanced_grid import AdvancedGrid

# Simulation components
from simulation.build_network import build_pandapower_network
from simulation.power_flow import PowerFlowEngine
from simulation.contingency_generator import (
    ContingencyGenerator, ScenarioFilter, load_contingencies
)
from simulation.contingency_analyzer import (
    ContingencyAnalyzer, screen_critical_contingencies
)
from simulation.mitigation_engine import (
    MitigationEngine, save_mitigation_plans
)

# ML components
try:
    from ml.ml_pipeline import MLPipeline, MLPipelineConfig
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML pipeline not available")


class BlendedTopologyOrchestrator:
    """
    Main orchestrator for the blended topology grid simulation system
    
    Coordinates all components: topology generation, contingency analysis,
    mitigation planning, and machine learning integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.grid = None
        self.base_net = None
        
        # Initialize output directories
        self._setup_output_directories()
        
        # Configure logging
        logger.add(
            f"outputs/logs/blended_topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level="INFO"
        )
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Grid configuration
            'grid': {
                'regions': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                'buses_per_region': 200,
                'target_size_buses': 2000
            },
            
            # Contingency configuration
            'contingency': {
                'include_n1': True,
                'include_n2': True,
                'max_n2_scenarios': 5000,
                'min_probability': 1e-8,
                'element_types': ['line', 'generator', 'transformer'],
                'analysis_parallel': True,
                'max_workers': 8
            },
            
            # Mitigation configuration
            'mitigation': {
                'generate_plans': True,
                'execute_plans': True,
                'economic_optimization': True,
                'voLL': 10000.0  # $/MWh
            },
            
            # ML configuration
            'ml': {
                'enable_training': True,
                'rf_enabled': True,
                'gnn_enabled': True,
                'rl_enabled': True,
                'training_data_fraction': 0.8
            },
            
            # Output configuration
            'output': {
                'save_topology': True,
                'save_scenarios': True,
                'save_results': True,
                'export_excel': True,
                'export_ml_dataset': True
            }
        }
    
    def _setup_output_directories(self):
        """Create necessary output directories"""
        directories = [
            'outputs/topology',
            'outputs/contingencies', 
            'outputs/analysis_results',
            'outputs/mitigation_plans',
            'outputs/ml_models',
            'outputs/logs',
            'outputs/excel_reports',
            'data/contingencies'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_full_system(self) -> Dict[str, Any]:
        """
        Run the complete blended topology system
        
        Returns:
            Summary of results and performance metrics
        """
        
        logger.info("🚀 Starting Blended Topology Grid Simulation System")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_time = time.time()
        results = {}
        
        try:
            # Phase 1: Grid Topology Generation
            logger.info("📊 Phase 1: Advanced Grid Topology Generation")
            topology_results = self._generate_advanced_topology()
            results['topology'] = topology_results
            
            # Phase 2: Contingency Generation
            logger.info("⚡ Phase 2: Contingency Scenario Generation")
            contingency_results = self._generate_contingency_scenarios()
            results['contingency_generation'] = contingency_results
            
            # Phase 3: Contingency Analysis
            logger.info("🔍 Phase 3: Comprehensive Contingency Analysis")
            analysis_results = self._analyze_contingencies()
            results['contingency_analysis'] = analysis_results
            
            # Phase 4: Mitigation Planning
            logger.info("🛠️ Phase 4: Mitigation Planning and Execution")
            mitigation_results = self._generate_mitigation_plans()
            results['mitigation'] = mitigation_results
            
            # Phase 5: ML Integration
            if ML_AVAILABLE and self.config['ml']['enable_training']:
                logger.info("🤖 Phase 5: Machine Learning Integration")
                ml_results = self._run_ml_pipeline()
                results['ml'] = ml_results
            else:
                logger.info("⏭️ Phase 5: ML Integration skipped")
                results['ml'] = {'skipped': True, 'reason': 'ML not available or disabled'}
            
            # Phase 6: Export and Reporting
            logger.info("📋 Phase 6: Export and Reporting")
            export_results = self._export_results()
            results['export'] = export_results
            
            total_time = time.time() - start_time
            
            # Final summary
            summary = self._generate_summary(results, total_time)
            results['summary'] = summary
            
            logger.info("✅ Blended Topology System completed successfully!")
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ System execution failed: {e}")
            results['error'] = str(e)
            results['execution_time'] = time.time() - start_time
            raise
    
    def _generate_advanced_topology(self) -> Dict[str, Any]:
        """Generate advanced grid topology with full component modeling"""
        
        start_time = time.time()
        
        # Create advanced grid with comprehensive modeling
        logger.info(f"Creating {self.config['grid']['target_size_buses']}-bus advanced grid...")
        
        self.grid = AdvancedGrid(
            regions=self.config['grid']['regions'],
            buses_per_region=self.config['grid']['buses_per_region']
        )
        
        # Build pandapower network for analysis
        logger.info("Building pandapower network for power flow analysis...")
        self.base_net, network_builder = build_pandapower_network(self.grid)
        
        # Verify power flow convergence
        logger.info("Verifying base case power flow convergence...")
        pf_engine = PowerFlowEngine()
        pf_result = pf_engine.solve_power_flow(self.base_net)
        
        # Save topology if configured
        if self.config['output']['save_topology']:
            topology_file = f"outputs/topology/advanced_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(topology_file, 'w') as f:
                f.write(self.grid.to_json())
            logger.info(f"Topology saved: {topology_file}")
        
        # Calculate topology metrics
        summary = self.grid.get_summary()
        
        results = {
            'grid_summary': summary,
            'base_case_converged': pf_result.converged,
            'base_case_iterations': pf_result.iterations,
            'base_case_solve_time': pf_result.solve_time_sec,
            'voltage_violations': len(pf_result.voltage_violations),
            'thermal_violations': len(pf_result.thermal_violations),
            'generation_time': time.time() - start_time
        }
        
        logger.info(f"✓ Topology generation completed: {summary['total_buses']} buses, "
                   f"{summary['total_generators']} generators, {summary['total_lines']} lines")
        
        return results
    
    def _generate_contingency_scenarios(self) -> Dict[str, Any]:
        """Generate comprehensive contingency scenarios"""
        
        start_time = time.time()
        
        # Create scenario filter
        scenario_filter = ScenarioFilter(
            include_n1=self.config['contingency']['include_n1'],
            include_n2=self.config['contingency']['include_n2'],
            max_n2_scenarios=self.config['contingency']['max_n2_scenarios'],
            min_probability=self.config['contingency']['min_probability'],
            element_types=self.config['contingency']['element_types']
        )
        
        # Generate scenarios
        logger.info("Generating N-1 and N-2 contingency scenarios...")
        generator = ContingencyGenerator(self.grid)
        metadata = generator.generate_all_scenarios(
            output_dir="data/contingencies",
            filters=scenario_filter
        )
        
        results = {
            'metadata': asdict(metadata),
            'generation_time': time.time() - start_time
        }
        
        logger.info(f"✓ Generated {metadata.total_scenarios} contingency scenarios "
                   f"(N-1: {metadata.n1_scenarios}, N-2: {metadata.n2_scenarios})")
        
        return results
    
    def _analyze_contingencies(self) -> Dict[str, Any]:
        """Perform comprehensive contingency analysis"""
        
        start_time = time.time()
        
        # Load contingency scenarios
        logger.info("Loading contingency scenarios...")
        contingencies = load_contingencies("data/contingencies")
        
        if not contingencies:
            logger.warning("No contingencies found - skipping analysis")
            return {'skipped': True, 'reason': 'No contingencies found'}
        
        logger.info(f"Loaded {len(contingencies)} contingency scenarios")
        
        # Analyze contingencies
        analyzer = ContingencyAnalyzer(self.grid)
        
        logger.info("Running contingency analysis (this may take several minutes)...")
        results = analyzer.analyze_all_contingencies(
            self.base_net,
            contingencies,
            parallel=self.config['contingency']['analysis_parallel'],
            max_workers=self.config['contingency']['max_workers']
        )
        
        # Screen critical contingencies
        critical_contingencies = screen_critical_contingencies(results, top_n=50)
        
        # Export results to DataFrame
        results_df = analyzer.export_results_to_dataframe(results)
        
        # Save analysis results
        results_file = f"outputs/analysis_results/contingency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Store results for later phases
        self.contingency_analysis_results = results
        self.critical_contingencies = critical_contingencies
        
        # Calculate summary statistics
        converged_count = sum(1 for r in results if r.converged)
        violation_count = sum(len(r.new_violations) for r in results)
        critical_count = len(critical_contingencies)
        
        analysis_results = {
            'total_scenarios_analyzed': len(results),
            'converged_scenarios': converged_count,
            'convergence_rate': converged_count / len(results) if results else 0,
            'total_new_violations': violation_count,
            'critical_contingencies': critical_count,
            'avg_solve_time': sum(r.solve_time for r in results) / len(results) if results else 0,
            'analysis_time': time.time() - start_time,
            'results_file': results_file
        }
        
        logger.info(f"✓ Contingency analysis completed: {converged_count}/{len(results)} converged, "
                   f"{critical_count} critical scenarios identified")
        
        return analysis_results
    
    def _generate_mitigation_plans(self) -> Dict[str, Any]:
        """Generate and execute mitigation plans"""
        
        start_time = time.time()
        
        if not hasattr(self, 'contingency_analysis_results'):
            logger.warning("No contingency analysis results - skipping mitigation")
            return {'skipped': True, 'reason': 'No contingency analysis results'}
        
        # Initialize mitigation engine
        engine = MitigationEngine(self.grid)
        
        # Generate mitigation plans for scenarios with violations
        logger.info("Generating mitigation plans for critical contingencies...")
        plans = []
        execution_results = []
        
        scenarios_with_violations = [r for r in self.contingency_analysis_results 
                                   if r.converged and r.new_violations]
        
        for i, result in enumerate(scenarios_with_violations[:100]):  # Limit for demonstration
            try:
                # Generate plan
                plan = engine.generate_mitigation_plan(result, self.base_net)
                
                if plan:
                    plans.append(plan)
                    
                    # Execute plan if configured
                    if self.config['mitigation']['execute_plans']:
                        execution_result = engine.execute_mitigation_plan(plan, self.base_net)
                        execution_results.append(execution_result)
                
                if (i + 1) % 25 == 0:
                    logger.info(f"Generated {i + 1}/{len(scenarios_with_violations)} mitigation plans")
                    
            except Exception as e:
                logger.warning(f"Error generating plan for {result.contingency.id}: {e}")
        
        # Save mitigation plans
        if plans:
            save_mitigation_plans(plans)
            
        # Store for ML pipeline
        self.mitigation_plans = plans
        self.execution_results = execution_results
        
        # Calculate summary statistics
        successful_executions = sum(1 for r in execution_results if r.success)
        total_cost = sum(r.final_cost_usd for r in execution_results)
        avg_violations_cleared = (sum(r.violations_cleared for r in execution_results) / 
                                len(execution_results) if execution_results else 0)
        
        mitigation_results = {
            'plans_generated': len(plans),
            'plans_executed': len(execution_results),
            'successful_executions': successful_executions,
            'execution_success_rate': successful_executions / len(execution_results) if execution_results else 0,
            'total_mitigation_cost_usd': total_cost,
            'avg_cost_per_plan_usd': total_cost / len(execution_results) if execution_results else 0,
            'avg_violations_cleared': avg_violations_cleared,
            'generation_time': time.time() - start_time
        }
        
        logger.info(f"✓ Mitigation planning completed: {len(plans)} plans generated, "
                   f"{successful_executions}/{len(execution_results)} successful executions")
        
        return mitigation_results
    
    def _run_ml_pipeline(self) -> Dict[str, Any]:
        """Run machine learning pipeline for mitigation learning"""
        
        start_time = time.time()
        
        if not hasattr(self, 'contingency_analysis_results') or not hasattr(self, 'mitigation_plans'):
            logger.warning("Insufficient data for ML training - skipping ML pipeline")
            return {'skipped': True, 'reason': 'Insufficient training data'}
        
        try:
            # Initialize ML pipeline
            ml_config = MLPipelineConfig()
            ml_pipeline = MLPipeline(self.grid, ml_config)
            
            # Filter data for training
            training_results = self.contingency_analysis_results[:int(
                len(self.contingency_analysis_results) * self.config['ml']['training_data_fraction']
            )]
            
            # Run full ML pipeline
            logger.info("Running ML pipeline: RF -> GNN -> RL")
            pipeline_results = ml_pipeline.run_full_pipeline(training_results, self.mitigation_plans)
            
            pipeline_results['training_time'] = time.time() - start_time
            
            logger.info("✓ ML pipeline completed successfully")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"ML pipeline failed: {e}")
            return {'error': str(e), 'training_time': time.time() - start_time}
    
    def _export_results(self) -> Dict[str, Any]:
        """Export comprehensive results and generate reports"""
        
        start_time = time.time()
        
        export_results = {}
        
        # Excel export
        if self.config['output']['export_excel']:
            excel_file = f"outputs/excel_reports/blended_topology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                
                # Grid summary
                if self.grid:
                    summary_df = pd.DataFrame([self.grid.get_summary()])
                    summary_df.to_excel(writer, sheet_name='Grid_Summary', index=False)
                
                # Contingency analysis results
                if hasattr(self, 'contingency_analysis_results'):
                    analyzer = ContingencyAnalyzer(self.grid)
                    analysis_df = analyzer.export_results_to_dataframe(self.contingency_analysis_results)
                    analysis_df.to_excel(writer, sheet_name='Contingency_Analysis', index=False)
                
                # Critical contingencies
                if hasattr(self, 'critical_contingencies'):
                    critical_df = analyzer.export_results_to_dataframe(self.critical_contingencies)
                    critical_df.to_excel(writer, sheet_name='Critical_Contingencies', index=False)
                
                # Mitigation plans summary
                if hasattr(self, 'mitigation_plans'):
                    plans_summary = []
                    for plan in self.mitigation_plans:
                        plans_summary.append({
                            'action_id': plan.action_id,
                            'contingency_id': plan.contingency_id,
                            'num_actions': len(plan.action_sequence),
                            'estimated_cost': sum(plan.cost_breakdown.values()),
                            'estimated_time': plan.execution_profile.get('estimated_time_s', 0)
                        })
                    
                    if plans_summary:
                        plans_df = pd.DataFrame(plans_summary)
                        plans_df.to_excel(writer, sheet_name='Mitigation_Plans', index=False)
            
            export_results['excel_file'] = excel_file
            logger.info(f"Excel report generated: {excel_file}")
        
        # ML dataset export
        if self.config['output']['export_ml_dataset'] and hasattr(self, 'contingency_analysis_results'):
            ml_dataset_file = f"outputs/ml_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Create comprehensive ML dataset
            analyzer = ContingencyAnalyzer(self.grid)
            ml_df = analyzer.export_results_to_dataframe(self.contingency_analysis_results)
            
            # Add mitigation features if available
            if hasattr(self, 'mitigation_plans'):
                mitigation_dict = {plan.contingency_id: plan for plan in self.mitigation_plans}
                
                ml_df['has_mitigation_plan'] = ml_df['contingency_id'].apply(
                    lambda x: x in mitigation_dict
                )
                
                ml_df['mitigation_type'] = ml_df['contingency_id'].apply(
                    lambda x: mitigation_dict[x].action_sequence[0].type 
                    if x in mitigation_dict and mitigation_dict[x].action_sequence 
                    else 'none'
                )
                
                ml_df['estimated_cost'] = ml_df['contingency_id'].apply(
                    lambda x: sum(mitigation_dict[x].cost_breakdown.values()) 
                    if x in mitigation_dict else 0
                )
            
            ml_df.to_csv(ml_dataset_file, index=False)
            export_results['ml_dataset_file'] = ml_dataset_file
            logger.info(f"ML dataset exported: {ml_dataset_file}")
        
        export_results['export_time'] = time.time() - start_time
        
        return export_results
    
    def _generate_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive system summary"""
        
        summary = {
            'system_info': {
                'total_execution_time': total_time,
                'completion_timestamp': datetime.now().isoformat(),
                'configuration': self.config
            },
            'topology_stats': results.get('topology', {}),
            'contingency_stats': {
                'scenarios_generated': results.get('contingency_generation', {}).get('metadata', {}).get('total_scenarios', 0),
                'scenarios_analyzed': results.get('contingency_analysis', {}).get('total_scenarios_analyzed', 0),
                'convergence_rate': results.get('contingency_analysis', {}).get('convergence_rate', 0),
                'critical_scenarios': results.get('contingency_analysis', {}).get('critical_contingencies', 0)
            },
            'mitigation_stats': {
                'plans_generated': results.get('mitigation', {}).get('plans_generated', 0),
                'execution_success_rate': results.get('mitigation', {}).get('execution_success_rate', 0),
                'total_cost_usd': results.get('mitigation', {}).get('total_mitigation_cost_usd', 0)
            },
            'ml_stats': results.get('ml', {}),
            'performance_metrics': {
                'topology_time': results.get('topology', {}).get('generation_time', 0),
                'contingency_time': results.get('contingency_analysis', {}).get('analysis_time', 0),
                'mitigation_time': results.get('mitigation', {}).get('generation_time', 0),
                'ml_time': results.get('ml', {}).get('training_time', 0),
                'export_time': results.get('export', {}).get('export_time', 0)
            }
        }
        
        # Save summary
        summary_file = f"outputs/system_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"System summary saved: {summary_file}")
        
        return summary


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Blended Topology Grid Simulation System')
    parser.add_argument('--mode', choices=['full', 'contingency_only', 'ml_only'], 
                       default='full', help='Execution mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--buses', type=int, default=2000, help='Target number of buses')
    parser.add_argument('--regions', type=int, default=10, help='Number of regions')
    parser.add_argument('--max-scenarios', type=int, default=5000, help='Maximum N-2 scenarios')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Update config with command line arguments
    if config is None:
        config = {}
    
    if 'grid' not in config:
        config['grid'] = {}
    
    config['grid']['target_size_buses'] = args.buses
    config['grid']['regions'] = [chr(65 + i) for i in range(args.regions)]  # A, B, C, ...
    config['grid']['buses_per_region'] = args.buses // args.regions
    
    if 'contingency' not in config:
        config['contingency'] = {}
    
    config['contingency']['max_n2_scenarios'] = args.max_scenarios
    config['contingency']['analysis_parallel'] = args.parallel
    
    if 'ml' not in config:
        config['ml'] = {}
    
    config['ml']['enable_training'] = not args.no_ml and ML_AVAILABLE
    
    # Create orchestrator and run
    orchestrator = BlendedTopologyOrchestrator(config)
    
    try:
        if args.mode == 'full':
            results = orchestrator.run_full_system()
        elif args.mode == 'contingency_only':
            # Run only topology and contingency analysis
            orchestrator._generate_advanced_topology()
            orchestrator._generate_contingency_scenarios()
            results = orchestrator._analyze_contingencies()
        elif args.mode == 'ml_only':
            # Assume data exists and run only ML
            results = orchestrator._run_ml_pipeline()
        
        print("\n" + "="*80)
        print("🎉 BLENDED TOPOLOGY SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"📊 Total Execution Time: {summary['system_info']['total_execution_time']:.2f} seconds")
            print(f"🏗️ Grid Size: {summary['topology_stats'].get('grid_summary', {}).get('total_buses', 0)} buses")
            print(f"⚡ Contingencies: {summary['contingency_stats']['scenarios_analyzed']} analyzed")
            print(f"🛠️ Mitigation Plans: {summary['mitigation_stats']['plans_generated']} generated")
            print(f"🤖 ML Training: {'✓' if summary['ml_stats'].get('rf_trained') else '✗'}")
        
        print("\nCheck outputs/ directory for detailed results and reports.")
        
    except Exception as e:
        print(f"\n❌ System execution failed: {e}")
        logger.error(f"System execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 