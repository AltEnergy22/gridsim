"""
Stressed Grid Simulation - Generate Violations and Mitigation Plans

Creates challenging scenarios specifically designed to force violations
and demonstrate mitigation planning capabilities.
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
from simulation.scenario_generator import ScenarioGenerator, ScenarioType
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

def create_stressed_scenarios(scenario_generator: ScenarioGenerator, count: int = 10):
    """Create scenarios specifically designed to cause violations"""
    
    scenarios = []
    
    # Force mostly stress and challenge scenarios
    for i in range(count):
        if i < 3:
            # Normal scenarios for baseline
            scenario = scenario_generator.generate_scenario(ScenarioType.NORMAL)
        elif i < 7:
            # Stress scenarios (high load)
            scenario = scenario_generator.generate_scenario(ScenarioType.STRESS)
        else:
            # Challenge scenarios (extreme conditions)
            scenario = scenario_generator.generate_scenario(ScenarioType.CHALLENGE)
        
        scenarios.append(scenario)
    
    return scenarios

def modify_network_for_stress(base_net):
    """Modify network to be more prone to violations"""
    
    # Reduce line capacities to create thermal stress
    for idx in base_net.line.index:
        current_max = base_net.line.at[idx, 'max_i_ka']
        base_net.line.at[idx, 'max_i_ka'] = current_max * 0.7  # Reduce by 30%
    
    # Reduce some generator capacity to create generation stress
    for idx in base_net.gen.index[::3]:  # Every 3rd generator
        current_max = base_net.gen.at[idx, 'max_p_mw']
        base_net.gen.at[idx, 'max_p_mw'] = current_max * 0.8  # Reduce by 20%
    
    return base_net

def main():
    """Run stressed simulation to generate violations and mitigation plans"""
    
    print_section("STRESSED GRID SIMULATION - FORCE VIOLATIONS & MITIGATION")
    print("Creating challenging conditions to demonstrate mitigation capabilities")
    
    # Configuration for stressed system
    config = {
        'grid_size': {
            'regions': ['A', 'B'],  
            'buses_per_region': 50,  # 100 total buses
        },
        'scenarios': {
            'count': 10,  
            'force_stress': True  # Force challenging scenarios
        },
        'contingencies': {
            'max_n1': 20,  # Focus on most critical 
            'max_n2': 10,   
        },
        'output_dir': 'outputs/stressed_simulation'
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # ================================================================
        # STEP 1: CREATE STRESSED GRID TOPOLOGY
        # ================================================================
        
        print_section("Step 1: Create Stressed Grid Topology")
        
        start_time = time.time()
        
        logger.info("Creating stressed grid with reduced margins...")
        
        grid = AdvancedGrid(
            regions=config['grid_size']['regions'],
            buses_per_region=config['grid_size']['buses_per_region']
        )
        
        # Build base network and stress it
        logger.info("Building and stressing pandapower network...")
        base_net, network_builder = build_pandapower_network(grid)
        
        # Stress the network to make violations more likely
        base_net = modify_network_for_stress(base_net)
        
        # Verify base case
        from simulation.power_flow import PowerFlowEngine
        pf_engine = PowerFlowEngine()
        base_result = pf_engine.solve_power_flow(base_net)
        
        if not base_result.converged:
            print("❌ Base case power flow failed to converge!")
            return
            
        print(f"✓ Stressed grid created: {len(base_net.bus)} buses, {len(base_net.line)} lines")
        print(f"✓ Base case converged: {base_result.total_generation_mw:.1f} MW generation")
        print(f"✓ Base violations: {len(base_result.voltage_violations)} voltage, {len(base_result.thermal_violations)} thermal")
        
        grid_time = time.time() - start_time
        
        # ================================================================
        # STEP 2: GENERATE STRESSED SCENARIOS  
        # ================================================================
        
        print_section("Step 2: Generate Stressed Operational Scenarios")
        
        start_time = time.time()
        
        # Generate mostly challenging scenarios
        scenario_generator = ScenarioGenerator()
        scenario_application = ScenarioApplication()
        
        logger.info(f"Generating {config['scenarios']['count']} stressed scenarios...")
        
        scenarios = create_stressed_scenarios(scenario_generator, config['scenarios']['count'])
        scenario_results = []
        
        for i, scenario in enumerate(scenarios):
            # Apply scenario to network
            scenario_net = scenario_application.apply_scenario(scenario, base_net)
            
            # Run power flow for this scenario
            pf_result = pf_engine.solve_power_flow(scenario_net)
            
            scenario_result = {
                'scenario_id': f"STRESS_{i+1:03d}",
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
        scenario_file = output_dir / f"stressed_scenarios_{timestamp}.csv"
        scenarios_df.to_csv(scenario_file, index=False)
        
        converged_scenarios = scenarios_df['converged'].sum()
        total_violations = scenarios_df['violations'].sum()
        print(f"✓ Generated {len(scenarios)} stressed scenarios")
        print(f"✓ Convergence rate: {converged_scenarios}/{len(scenarios)} ({converged_scenarios/len(scenarios)*100:.1f}%)")
        print(f"✓ Total violations across scenarios: {total_violations}")
        print(f"✓ Scenario dataset: {scenario_file}")
        
        scenario_time = time.time() - start_time
        
        # ================================================================
        # STEP 3: GENERATE CRITICAL CONTINGENCIES
        # ================================================================
        
        print_section("Step 3: Generate Critical Contingency Scenarios")
        
        start_time = time.time()
        
        # Focus on most critical contingencies
        scenario_filter = ScenarioFilter(
            include_n1=True,
            include_n2=True,
            max_n2_scenarios=config['contingencies']['max_n2'],
            min_probability=1e-5,  # Include more scenarios
            element_types=['line', 'generator']
        )
        
        logger.info("Generating critical contingency scenarios...")
        contingency_generator = ContingencyGenerator(grid)
        
        contingency_dir = output_dir / "contingencies"
        metadata = contingency_generator.generate_all_scenarios(
            output_dir=str(contingency_dir),
            filters=scenario_filter
        )
        
        # Load generated contingencies (take most probable ones)
        from simulation.contingency_generator import load_contingencies
        all_contingencies = load_contingencies(str(contingency_dir))
        
        # Sort by probability (most likely failures first) and take subset
        all_contingencies.sort(key=lambda x: x.probability, reverse=True)
        demo_contingencies = all_contingencies[:config['contingencies']['max_n1']]
        
        print(f"✓ Generated {metadata.total_scenarios} total contingency scenarios")
        print(f"✓ Using {len(demo_contingencies)} most critical scenarios for analysis")
        
        contingency_gen_time = time.time() - start_time
        
        # ================================================================
        # STEP 4: STRESSED CONTINGENCY ANALYSIS
        # ================================================================
        
        print_section("Step 4: Stressed Contingency Analysis")
        
        start_time = time.time()
        
        # Use the most stressed scenario for contingency analysis
        stress_scenarios = [s for s in scenarios if s.scenario_type == ScenarioType.STRESS or s.scenario_type == ScenarioType.CHALLENGE]
        if stress_scenarios:
            test_scenario = stress_scenarios[0]  # Use first stress scenario
            logger.info(f"Using stressed scenario: load_mult={test_scenario.load_multiplier:.3f}, temp={test_scenario.weather_temp_c:.1f}°C")
            
            # Apply stress scenario to base network for contingency analysis
            stressed_net = scenario_application.apply_scenario(test_scenario, base_net)
        else:
            stressed_net = base_net
            logger.info("Using base network for contingency analysis")
        
        logger.info(f"Analyzing {len(demo_contingencies)} contingency scenarios on stressed network...")
        
        analyzer = ContingencyAnalyzer(grid)
        contingency_results = analyzer.analyze_all_contingencies(
            stressed_net, demo_contingencies, parallel=True, max_workers=4
        )
        
        # Export contingency analysis results
        analysis_df = analyzer.export_results_to_dataframe(contingency_results)
        contingency_file = output_dir / f"stressed_contingency_analysis_{timestamp}.csv"
        analysis_df.to_csv(contingency_file, index=False)
        
        # Statistics
        converged_contingencies = sum(1 for r in contingency_results if r.converged)
        with_violations = sum(1 for r in contingency_results if r.converged and r.new_violations)
        critical_contingencies = sum(1 for r in contingency_results if r.criticality_level in ['critical', 'high'])
        
        print(f"✓ Analyzed {len(contingency_results)} contingency scenarios")
        print(f"✓ Converged: {converged_contingencies}/{len(contingency_results)} ({converged_contingencies/len(contingency_results)*100:.1f}%)")
        print(f"✓ With NEW violations: {with_violations}")
        print(f"✓ Critical/High severity: {critical_contingencies}")
        print(f"✓ Contingency dataset: {contingency_file}")
        
        analysis_time = time.time() - start_time
        
        # ================================================================
        # STEP 5: COMPREHENSIVE MITIGATION PLANNING
        # ================================================================
        
        print_section("Step 5: Comprehensive Mitigation Planning")
        
        start_time = time.time()
        
        # Generate mitigation plans for ALL scenarios with violations (not just new ones)
        logger.info("Generating mitigation plans for scenarios with violations...")
        
        mitigation_engine = MitigationEngine(grid)
        mitigation_plans = []
        execution_results = []
        
        # Include scenarios with any violations (not just new ones)
        violation_scenarios = [r for r in contingency_results if r.converged and (r.new_violations or r.post_violations)]
        
        logger.info(f"Found {len(violation_scenarios)} scenarios with violations for mitigation planning")
        
        for i, result in enumerate(violation_scenarios):
            try:
                # Generate mitigation plan
                plan = mitigation_engine.generate_mitigation_plan(result, stressed_net)
                
                if plan:
                    mitigation_plans.append(plan)
                    
                    # Execute plan to validate effectiveness
                    execution_result = mitigation_engine.execute_mitigation_plan(plan, stressed_net)
                    execution_results.append(execution_result)
                    
                    logger.info(f"Generated plan {plan.action_id}: {len(plan.action_sequence)} actions, success={execution_result.success}")
                    
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
                    'primary_target': plan.action_sequence[0].target if plan.action_sequence else 'none',
                    'primary_delta_mw': plan.action_sequence[0].parameters.get('delta_mw', 0) if plan.action_sequence else 0,
                    'estimated_cost_usd': sum(plan.cost_breakdown.values()),
                    'execution_success': execution.success,
                    'violations_cleared': execution.violations_cleared,
                    'violations_remaining': execution.violations_remaining,
                    'final_cost_usd': execution.final_cost_usd,
                    'execution_time_s': execution.execution_time_s,
                    'total_mw_redispatched': execution.total_mw_redispatched,
                    'total_mw_shed': execution.total_mw_shed
                }
                plans_summary.append(summary)
            
            plans_df = pd.DataFrame(plans_summary)
            plans_file = output_dir / f"mitigation_plans_{timestamp}.csv"
            plans_df.to_csv(plans_file, index=False)
            
            # Statistics
            successful_plans = sum(1 for r in execution_results if r.success)
            total_cost = sum(r.final_cost_usd for r in execution_results)
            total_violations_cleared = sum(r.violations_cleared for r in execution_results)
            total_redispatch = sum(r.total_mw_redispatched for r in execution_results)
            total_shed = sum(r.total_mw_shed for r in execution_results)
            
            print(f"✓ Generated {len(mitigation_plans)} mitigation plans")
            print(f"✓ Successful executions: {successful_plans}/{len(execution_results)} ({successful_plans/len(execution_results)*100:.1f}%)")
            print(f"✓ Total violations cleared: {total_violations_cleared}")
            print(f"✓ Total mitigation cost: ${total_cost:,.2f}")
            print(f"✓ Total redispatch: {total_redispatch:.1f} MW")
            print(f"✓ Total load shed: {total_shed:.1f} MW")
            print(f"✓ Mitigation dataset: {plans_file}")
        else:
            print("✗ No mitigation plans generated - no actionable violations found")
        
        mitigation_time = time.time() - start_time
        
        # ================================================================
        # STEP 6: COMPREHENSIVE DATASET EXPORT
        # ================================================================
        
        print_section("Step 6: Comprehensive Dataset Export")
        
        # Create master Excel file with all datasets
        excel_file = output_dir / f"stressed_simulation_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Grid summary
            grid_summary = pd.DataFrame([{
                'total_buses': len(base_net.bus),
                'total_lines': len(base_net.line),
                'total_generators': len(base_net.gen),
                'base_generation_mw': base_result.total_generation_mw,
                'base_load_mw': base_result.total_load_mw,
                'base_violations': len(base_result.voltage_violations) + len(base_result.thermal_violations)
            }])
            grid_summary.to_excel(writer, sheet_name='Grid_Summary', index=False)
            
            # Stressed scenarios
            scenarios_df.to_excel(writer, sheet_name='Stressed_Scenarios', index=False)
            
            # Contingency analysis
            analysis_df.to_excel(writer, sheet_name='Contingency_Analysis', index=False)
            
            # Mitigation plans
            if mitigation_plans:
                plans_df.to_excel(writer, sheet_name='Mitigation_Plans', index=False)
        
        print(f"✓ Comprehensive Excel report: {excel_file}")
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        
        print_section("STRESSED SIMULATION COMPLETED SUCCESSFULLY!")
        
        total_time = grid_time + scenario_time + contingency_gen_time + analysis_time + mitigation_time
        
        print(f"📊 Execution Summary:")
        print(f"  • Total time: {total_time:.2f} seconds")
        print(f"  • Grid generation: {grid_time:.2f}s")
        print(f"  • Scenario generation: {scenario_time:.2f}s") 
        print(f"  • Contingency generation: {contingency_gen_time:.2f}s")
        print(f"  • Contingency analysis: {analysis_time:.2f}s")
        print(f"  • Mitigation planning: {mitigation_time:.2f}s")
        
        print(f"\n📁 Generated Datasets:")
        print(f"  • Stressed scenarios: {scenario_file}")
        print(f"  • Contingency analysis: {contingency_file}")
        if mitigation_plans:
            print(f"  • Mitigation plans: {plans_file}")
        print(f"  • Comprehensive Excel: {excel_file}")
        
        print(f"\n🎯 Key Results:")
        print(f"  • Grid: {len(base_net.bus)} buses, {len(base_net.line)} lines (stressed)")
        print(f"  • Scenarios: {len(scenarios)} stressed scenarios")
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
        print("\n🎉 Stressed simulation completed successfully!")
        print("Check the outputs/stressed_simulation/ directory for datasets with mitigation plans.")
    else:
        print("\n💥 Simulation failed. Check error messages above.")
        exit(1) 