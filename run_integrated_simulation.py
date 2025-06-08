"""
Integrated Simulation System

Combines multiple simulation components to generate comprehensive datasets:
1. Realistic Simulation Orchestrator - Generate operational scenarios
2. Contingency Generator - Create N-1/N-2 scenarios  
3. Contingency Analyzer - Analyze impacts and violations
4. Mitigation Engine - Generate action plans
5. Export comprehensive datasets for analysis

This creates the datasets you requested to examine.
"""

import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Core grid components
from grid.advanced_grid import AdvancedGrid
from simulation.build_network import build_pandapower_network

# Simulation orchestration  
from simulation.realistic_simulation_orchestrator import RealisticSimulationOrchestrator
from simulation.scenario_generator import ScenarioGenerator
from simulation.scenario_application import ScenarioApplication

# Contingency analysis
from simulation.contingency_generator import ContingencyGenerator, ScenarioFilter
from simulation.contingency_analyzer import ContingencyAnalyzer

# Mitigation planning
from simulation.mitigation_engine import MitigationEngine, save_mitigation_plans

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print('='*60)

def main():
    """Run integrated simulation to generate comprehensive datasets"""
    
    print_section("INTEGRATED SIMULATION SYSTEM")
    print("Combining realistic scenarios + contingency analysis + mitigation planning")
    
    # Configuration for manageable demonstration
    config = {
        'grid_size': {
            'regions': ['A', 'B'],  # Start with 2 regions  
            'buses_per_region': 50,  # 100 total buses
        },
        'scenarios': {
            'count': 10,  # Generate 10 realistic scenarios
            'types': ['normal', 'stress', 'challenge']
        },
        'contingencies': {
            'max_n1': 30,  # Limit N-1 scenarios
            'max_n2': 20,  # Limit N-2 scenarios  
        },
        'output_dir': 'outputs/integrated_simulation'
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # ================================================================
        # STEP 1: CREATE BASE GRID TOPOLOGY
        # ================================================================
        
        print_section("Step 1: Create Base Grid Topology")
        
        start_time = time.time()
        
        # Create manageable grid for demonstration
        logger.info(f"Creating {sum(config['grid_size']['buses_per_region'] for _ in config['grid_size']['regions'])}-bus grid...")
        
        grid = AdvancedGrid(
            regions=config['grid_size']['regions'],
            buses_per_region=config['grid_size']['buses_per_region']
        )
        
        # Build base network
        logger.info("Building pandapower network...")
        base_net, network_builder = build_pandapower_network(grid)
        
        # Verify base case
        from simulation.power_flow import PowerFlowEngine
        pf_engine = PowerFlowEngine()
        base_result = pf_engine.solve_power_flow(base_net)
        
        if not base_result.converged:
            print("❌ Base case power flow failed to converge!")
            return
            
        print(f"✓ Base grid created: {len(base_net.bus)} buses, {len(base_net.line)} lines")
        print(f"✓ Base case converged: {base_result.total_generation_mw:.1f} MW generation")
        
        grid_time = time.time() - start_time
        
        # ================================================================
        # STEP 2: GENERATE REALISTIC OPERATIONAL SCENARIOS  
        # ================================================================
        
        print_section("Step 2: Generate Realistic Operational Scenarios")
        
        start_time = time.time()
        
        # Initialize scenario components
        scenario_generator = ScenarioGenerator()
        scenario_application = ScenarioApplication()
        
        # Generate diverse scenarios
        logger.info(f"Generating {config['scenarios']['count']} realistic scenarios...")
        
        scenarios = []
        scenario_results = []
        
        for i in range(config['scenarios']['count']):
            # Generate scenario parameters (random type selection)
            scenario = scenario_generator.generate_scenario()
            scenarios.append(scenario)
            
            # Apply scenario to network
            scenario_net = scenario_application.apply_scenario(scenario, base_net)
            
            # Run power flow for this scenario
            pf_result = pf_engine.solve_power_flow(scenario_net)
            
            scenario_result = {
                'scenario_id': f"SCEN_{i+1:03d}",
                'scenario_type': scenario.scenario_type.value,
                'load_multiplier': scenario.load_multiplier,
                'renewable_cf_wind': scenario.renewable_cf.get('wind', 0),
                'renewable_cf_solar': scenario.renewable_cf.get('solar', 0),
                'temperature_c': scenario.weather_temp_c,
                'converged': pf_result.converged,
                'total_generation_mw': pf_result.total_generation_mw if pf_result.converged else 0,
                'total_load_mw': pf_result.total_load_mw if pf_result.converged else 0,
                'max_voltage_pu': pf_result.max_voltage_pu if pf_result.converged else 0,
                'min_voltage_pu': pf_result.min_voltage_pu if pf_result.converged else 0,
                'violations': len(pf_result.voltage_violations) + len(pf_result.thermal_violations) if pf_result.converged else 0
            }
            
            scenario_results.append(scenario_result)
            
            if (i + 1) % 3 == 0:
                logger.info(f"Generated {i + 1}/{config['scenarios']['count']} scenarios")
        
        # Export scenario dataset
        scenarios_df = pd.DataFrame(scenario_results)
        scenario_file = output_dir / f"realistic_scenarios_{timestamp}.csv"
        scenarios_df.to_csv(scenario_file, index=False)
        
        converged_scenarios = scenarios_df['converged'].sum()
        print(f"✓ Generated {len(scenarios)} operational scenarios")
        print(f"✓ Convergence rate: {converged_scenarios}/{len(scenarios)} ({converged_scenarios/len(scenarios)*100:.1f}%)")
        print(f"✓ Scenario dataset: {scenario_file}")
        
        scenario_time = time.time() - start_time
        
        # ================================================================
        # STEP 3: GENERATE CONTINGENCY SCENARIOS
        # ================================================================
        
        print_section("Step 3: Generate Contingency Scenarios")
        
        start_time = time.time()
        
        # Create contingency filter for manageable set
        scenario_filter = ScenarioFilter(
            include_n1=True,
            include_n2=True,
            max_n2_scenarios=config['contingencies']['max_n2'],
            min_probability=1e-6,
            element_types=['line', 'generator']  # Focus on lines and generators
        )
        
        # Generate contingency scenarios
        logger.info("Generating N-1 and N-2 contingency scenarios...")
        contingency_generator = ContingencyGenerator(grid)
        
        contingency_dir = output_dir / "contingencies"
        metadata = contingency_generator.generate_all_scenarios(
            output_dir=str(contingency_dir),
            filters=scenario_filter
        )
        
        # Load generated contingencies (limit for demo)
        from simulation.contingency_generator import load_contingencies
        all_contingencies = load_contingencies(str(contingency_dir))
        
        # Take subset for demonstration
        demo_contingencies = all_contingencies[:config['contingencies']['max_n1']]
        
        print(f"✓ Generated {metadata.total_scenarios} total contingency scenarios")
        print(f"✓ Using {len(demo_contingencies)} scenarios for analysis")
        print(f"✓ N-1 scenarios: {metadata.n1_scenarios}")
        print(f"✓ N-2 scenarios: {metadata.n2_scenarios}")
        
        contingency_gen_time = time.time() - start_time
        
        # ================================================================
        # STEP 4: CONTINGENCY ANALYSIS
        # ================================================================
        
        print_section("Step 4: Contingency Analysis")
        
        start_time = time.time()
        
        # Run contingency analysis
        logger.info(f"Analyzing {len(demo_contingencies)} contingency scenarios...")
        
        analyzer = ContingencyAnalyzer(grid)
        contingency_results = analyzer.analyze_all_contingencies(
            base_net, demo_contingencies, parallel=True, max_workers=4
        )
        
        # Export contingency analysis results
        analysis_df = analyzer.export_results_to_dataframe(contingency_results)
        contingency_file = output_dir / f"contingency_analysis_{timestamp}.csv"
        analysis_df.to_csv(contingency_file, index=False)
        
        # Statistics
        converged_contingencies = sum(1 for r in contingency_results if r.converged)
        with_violations = sum(1 for r in contingency_results if r.converged and r.new_violations)
        critical_contingencies = sum(1 for r in contingency_results if r.criticality_level in ['critical', 'high'])
        
        print(f"✓ Analyzed {len(contingency_results)} contingency scenarios")
        print(f"✓ Converged: {converged_contingencies}/{len(contingency_results)} ({converged_contingencies/len(contingency_results)*100:.1f}%)")
        print(f"✓ With new violations: {with_violations}")
        print(f"✓ Critical/High severity: {critical_contingencies}")
        print(f"✓ Contingency dataset: {contingency_file}")
        
        analysis_time = time.time() - start_time
        
        # ================================================================
        # STEP 5: MITIGATION PLANNING
        # ================================================================
        
        print_section("Step 5: Mitigation Planning")
        
        start_time = time.time()
        
        # Generate mitigation plans for scenarios with violations
        logger.info("Generating mitigation plans for scenarios with violations...")
        
        mitigation_engine = MitigationEngine(grid)
        mitigation_plans = []
        execution_results = []
        
        violation_scenarios = [r for r in contingency_results if r.converged and r.new_violations]
        
        for i, result in enumerate(violation_scenarios):
            try:
                # Generate mitigation plan
                plan = mitigation_engine.generate_mitigation_plan(result, base_net)
                
                if plan:
                    mitigation_plans.append(plan)
                    
                    # Execute plan to validate effectiveness
                    execution_result = mitigation_engine.execute_mitigation_plan(plan, base_net)
                    execution_results.append(execution_result)
                    
                if (i + 1) % 5 == 0:
                    logger.info(f"Generated {i + 1}/{len(violation_scenarios)} mitigation plans")
                    
            except Exception as e:
                logger.warning(f"Error generating plan for {result.contingency.id}: {e}")
        
        # Export mitigation plans
        if mitigation_plans:
            # Save individual plans as JSON
            plans_dir = output_dir / "mitigation_plans"
            save_mitigation_plans(mitigation_plans, str(plans_dir))
            
            # Create summary dataset
            plans_summary = []
            for plan, execution in zip(mitigation_plans, execution_results):
                summary = {
                    'action_id': plan.action_id,
                    'contingency_id': plan.contingency_id,
                    'num_actions': len(plan.action_sequence),
                    'primary_action_type': plan.action_sequence[0].type if plan.action_sequence else 'none',
                    'estimated_cost_usd': sum(plan.cost_breakdown.values()),
                    'execution_success': execution.success,
                    'violations_cleared': execution.violations_cleared,
                    'final_cost_usd': execution.final_cost_usd,
                    'execution_time_s': execution.execution_time_s
                }
                plans_summary.append(summary)
            
            plans_df = pd.DataFrame(plans_summary)
            plans_file = output_dir / f"mitigation_plans_{timestamp}.csv"
            plans_df.to_csv(plans_file, index=False)
            
            # Statistics
            successful_plans = sum(1 for r in execution_results if r.success)
            total_cost = sum(r.final_cost_usd for r in execution_results)
            total_violations_cleared = sum(r.violations_cleared for r in execution_results)
            
            print(f"✓ Generated {len(mitigation_plans)} mitigation plans")
            print(f"✓ Successful executions: {successful_plans}/{len(execution_results)} ({successful_plans/len(execution_results)*100:.1f}%)")
            print(f"✓ Total violations cleared: {total_violations_cleared}")
            print(f"✓ Total mitigation cost: ${total_cost:,.2f}")
            print(f"✓ Mitigation dataset: {plans_file}")
        
        mitigation_time = time.time() - start_time
        
        # ================================================================
        # STEP 6: COMPREHENSIVE DATASET EXPORT
        # ================================================================
        
        print_section("Step 6: Comprehensive Dataset Export")
        
        # Create master Excel file with all datasets
        excel_file = output_dir / f"integrated_simulation_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Grid summary
            grid_summary = pd.DataFrame([{
                'total_buses': len(base_net.bus),
                'total_lines': len(base_net.line),
                'total_generators': len(base_net.gen),
                'base_generation_mw': base_result.total_generation_mw,
                'base_load_mw': base_result.total_load_mw,
                'base_losses_mw': base_result.total_losses_mw
            }])
            grid_summary.to_excel(writer, sheet_name='Grid_Summary', index=False)
            
            # Realistic scenarios
            scenarios_df.to_excel(writer, sheet_name='Realistic_Scenarios', index=False)
            
            # Contingency analysis
            analysis_df.to_excel(writer, sheet_name='Contingency_Analysis', index=False)
            
            # Mitigation plans
            if mitigation_plans:
                plans_df.to_excel(writer, sheet_name='Mitigation_Plans', index=False)
        
        # Create combined dataset for ML
        if mitigation_plans:
            # Merge contingency analysis with mitigation results
            mitigation_dict = {plan.contingency_id: plan for plan in mitigation_plans}
            
            # Add mitigation features to contingency analysis
            analysis_df['has_mitigation_plan'] = analysis_df['contingency_id'].apply(
                lambda x: x in mitigation_dict
            )
            
            analysis_df['mitigation_type'] = analysis_df['contingency_id'].apply(
                lambda x: mitigation_dict[x].action_sequence[0].type 
                if x in mitigation_dict and mitigation_dict[x].action_sequence 
                else 'none'
            )
            
            analysis_df['mitigation_cost'] = analysis_df['contingency_id'].apply(
                lambda x: sum(mitigation_dict[x].cost_breakdown.values()) 
                if x in mitigation_dict else 0
            )
            
            # Export ML-ready dataset
            ml_dataset_file = output_dir / f"ml_ready_dataset_{timestamp}.csv"
            analysis_df.to_csv(ml_dataset_file, index=False)
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        
        print_section("SIMULATION COMPLETED SUCCESSFULLY!")
        
        total_time = grid_time + scenario_time + contingency_gen_time + analysis_time + mitigation_time
        
        print(f"📊 Execution Summary:")
        print(f"  • Total time: {total_time:.2f} seconds")
        print(f"  • Grid generation: {grid_time:.2f}s")
        print(f"  • Scenario generation: {scenario_time:.2f}s") 
        print(f"  • Contingency generation: {contingency_gen_time:.2f}s")
        print(f"  • Contingency analysis: {analysis_time:.2f}s")
        print(f"  • Mitigation planning: {mitigation_time:.2f}s")
        
        print(f"\n📁 Generated Datasets:")
        print(f"  • Realistic scenarios: {scenario_file}")
        print(f"  • Contingency analysis: {contingency_file}")
        if mitigation_plans:
            print(f"  • Mitigation plans: {plans_file}")
            print(f"  • ML-ready dataset: {ml_dataset_file}")
        print(f"  • Comprehensive Excel: {excel_file}")
        
        print(f"\n🎯 Key Results:")
        print(f"  • Grid: {len(base_net.bus)} buses, {len(base_net.line)} lines")
        print(f"  • Scenarios: {len(scenarios)} operational scenarios")
        print(f"  • Contingencies: {len(contingency_results)} analyzed ({converged_contingencies} converged)")
        print(f"  • Violations: {with_violations} scenarios with new violations")
        print(f"  • Mitigation: {len(mitigation_plans)} plans generated")
        
        print(f"\n✅ All datasets available in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Integrated simulation completed successfully!")
        print("Check the outputs/integrated_simulation/ directory for all datasets.")
    else:
        print("\n💥 Simulation failed. Check error messages above.")
        exit(1) 