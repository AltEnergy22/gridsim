"""
Mitigation Demo - Force Generation of Mitigation Plans

Creates a targeted demo that forces violations and mitigation plan generation
by temporarily modifying the mitigation logic.
"""

import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import copy

from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Core grid components
from grid.advanced_grid import AdvancedGrid
from simulation.build_network import build_pandapower_network

# Simulation orchestration  
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


class ForcedMitigationEngine(MitigationEngine):
    """Modified mitigation engine that generates plans for ALL violations"""
    
    def generate_mitigation_plan(self, contingency_result, base_net):
        """Generate mitigation plan - considers ALL violations not just new ones"""
        
        # Get all violations (pre + post + new)
        all_violations = []
        
        # Add all post-contingency violations 
        if hasattr(contingency_result, 'post_violations') and contingency_result.post_violations:
            all_violations.extend(contingency_result.post_violations)
        
        # Add new violations 
        if hasattr(contingency_result, 'new_violations') and contingency_result.new_violations:
            all_violations.extend(contingency_result.new_violations)
        
        # If still no violations, skip
        if not all_violations:
            return None
        
        logger.info(f"Generating mitigation plan for contingency {contingency_result.contingency.id} with {len(all_violations)} total violations")
        
        # Create plan ID
        action_id = f"ACT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{contingency_result.contingency.id}"
        
        from simulation.mitigation_engine import MitigationPlan, ActionStep
        
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
        voltage_violations = [v for v in all_violations if 'voltage' in v.violation_type]
        thermal_violations = [v for v in all_violations if 'thermal' in v.violation_type]
        
        logger.info(f"Found {len(voltage_violations)} voltage violations and {len(thermal_violations)} thermal violations")
        
        # Handle thermal violations first (usually more critical)
        for violation in thermal_violations[:5]:  # Limit to first 5 to avoid too many actions
            step, cost = self._create_thermal_mitigation_step(violation, step_counter, base_net)
            if step:
                plan.action_sequence.append(step)
                total_cost += cost
                step_counter += 1
        
        # Handle voltage violations
        for violation in voltage_violations[:5]:  # Limit to first 5 to avoid too many actions  
            step, cost = self._create_voltage_mitigation_step(violation, step_counter, base_net)
            if step:
                plan.action_sequence.append(step)
                total_cost += cost
                step_counter += 1
        
        # If no actionable violations found, create a generic plan
        if not plan.action_sequence:
            # Create a generic redispatch action
            logger.info("No specific violations found - creating generic redispatch plan")
            
            generic_step = ActionStep(
                step=1,
                type='redispatch',
                target='GEN_A001',  # Use first generator 
                parameters={
                    'delta_mw': 10.0,
                    'ramp_rate_mw_per_min': 5.0,
                    'reason': f'Preventive redispatch for contingency {contingency_result.contingency.id}'
                }
            )
            plan.action_sequence.append(generic_step)
            total_cost = 150.0  # Estimated cost
        
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
        
        logger.info(f"Generated plan with {len(plan.action_sequence)} action steps")
        
        return plan


def main():
    """Run targeted mitigation demo"""
    
    print_section("MITIGATION PLAN GENERATION DEMO")
    print("Targeted demo to force generation of mitigation action plans")
    
    # Configuration for demo
    config = {
        'grid_size': {
            'regions': ['A', 'B'],  
            'buses_per_region': 30,  # Smaller for focused demo
        },
        'scenarios': {
            'count': 5,  # Just a few scenarios
        },
        'contingencies': {
            'max_n1': 10,  # Focus on most critical 
        },
        'output_dir': 'outputs/mitigation_demo'
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # ================================================================
        # STEP 1: CREATE DEMO GRID
        # ================================================================
        
        print_section("Step 1: Create Demo Grid")
        
        start_time = time.time()
        
        logger.info("Creating demo grid for mitigation testing...")
        
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
            
        print(f"✓ Demo grid created: {len(base_net.bus)} buses, {len(base_net.line)} lines")
        print(f"✓ Base case converged: {base_result.total_generation_mw:.1f} MW generation")
        print(f"✓ Base violations: {len(base_result.voltage_violations)} voltage, {len(base_result.thermal_violations)} thermal")
        
        grid_time = time.time() - start_time
        
        # ================================================================
        # STEP 2: GENERATE STRESSED SCENARIO
        # ================================================================
        
        print_section("Step 2: Generate Stressed Scenario")
        
        start_time = time.time()
        
        # Generate one high-stress scenario
        scenario_generator = ScenarioGenerator()
        scenario_application = ScenarioApplication()
        
        stress_scenario = scenario_generator.generate_scenario(ScenarioType.CHALLENGE)
        
        logger.info(f"Generated stress scenario: load_mult={stress_scenario.load_multiplier:.3f}, temp={stress_scenario.weather_temp_c:.1f}°C")
        
        # Apply scenario to network
        stressed_net = scenario_application.apply_scenario(stress_scenario, base_net)
        
        # Run power flow for stressed scenario
        stressed_result = pf_engine.solve_power_flow(stressed_net)
        
        if not stressed_result.converged:
            print("❌ Stressed scenario power flow failed!")
            return
            
        print(f"✓ Stressed scenario converged: {stressed_result.total_generation_mw:.1f} MW generation")
        print(f"✓ Stressed violations: {len(stressed_result.voltage_violations)} voltage, {len(stressed_result.thermal_violations)} thermal")
        print(f"✓ Total violations in stressed case: {len(stressed_result.voltage_violations) + len(stressed_result.thermal_violations)}")
        
        scenario_time = time.time() - start_time
        
        # ================================================================
        # STEP 3: GENERATE CRITICAL CONTINGENCIES
        # ================================================================
        
        print_section("Step 3: Generate Critical Contingencies")
        
        start_time = time.time()
        
        # Focus on most critical contingencies
        scenario_filter = ScenarioFilter(
            include_n1=True,
            include_n2=False,  # Skip N-2 for demo simplicity
            min_probability=1e-4,
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
        
        # Sort by probability and take subset
        all_contingencies.sort(key=lambda x: x.probability, reverse=True)
        demo_contingencies = all_contingencies[:config['contingencies']['max_n1']]
        
        print(f"✓ Generated {metadata.total_scenarios} total contingency scenarios")
        print(f"✓ Using {len(demo_contingencies)} most critical scenarios for analysis")
        
        contingency_gen_time = time.time() - start_time
        
        # ================================================================
        # STEP 4: CONTINGENCY ANALYSIS ON STRESSED NETWORK
        # ================================================================
        
        print_section("Step 4: Contingency Analysis on Stressed Network")
        
        start_time = time.time()
        
        logger.info(f"Analyzing {len(demo_contingencies)} contingency scenarios on stressed network...")
        
        analyzer = ContingencyAnalyzer(grid)
        contingency_results = analyzer.analyze_all_contingencies(
            stressed_net, demo_contingencies, parallel=False, max_workers=1  # Sequential for demo
        )
        
        # Export contingency analysis results
        analysis_df = analyzer.export_results_to_dataframe(contingency_results)
        contingency_file = output_dir / f"contingency_analysis_{timestamp}.csv"
        analysis_df.to_csv(contingency_file, index=False)
        
        # Statistics
        converged_contingencies = sum(1 for r in contingency_results if r.converged)
        with_violations = sum(1 for r in contingency_results if r.converged and (r.new_violations or r.post_violations))
        
        print(f"✓ Analyzed {len(contingency_results)} contingency scenarios")
        print(f"✓ Converged: {converged_contingencies}/{len(contingency_results)} ({converged_contingencies/len(contingency_results)*100:.1f}%)")
        print(f"✓ With violations: {with_violations}")
        print(f"✓ Contingency dataset: {contingency_file}")
        
        analysis_time = time.time() - start_time
        
        # ================================================================
        # STEP 5: FORCED MITIGATION PLANNING
        # ================================================================
        
        print_section("Step 5: Forced Mitigation Planning")
        
        start_time = time.time()
        
        # Use our modified mitigation engine that considers ALL violations
        logger.info("Generating mitigation plans using modified engine...")
        
        mitigation_engine = ForcedMitigationEngine(grid)
        mitigation_plans = []
        execution_results = []
        
        # Generate plans for converged scenarios (even without new violations)
        viable_scenarios = [r for r in contingency_results if r.converged]
        
        logger.info(f"Generating mitigation plans for {len(viable_scenarios)} viable scenarios")
        
        for i, result in enumerate(viable_scenarios):
            try:
                # Generate mitigation plan (forced)
                plan = mitigation_engine.generate_mitigation_plan(result, stressed_net)
                
                if plan:
                    mitigation_plans.append(plan)
                    
                    # Execute plan to validate effectiveness
                    execution_result = mitigation_engine.execute_mitigation_plan(plan, stressed_net)
                    execution_results.append(execution_result)
                    
                    logger.info(f"Generated plan {plan.action_id}: {len(plan.action_sequence)} actions, success={execution_result.success}")
                    
                if (i + 1) % 3 == 0:
                    logger.info(f"Generated {i + 1}/{len(viable_scenarios)} mitigation plans")
                    
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
                    'estimated_cost_usd': sum(plan.cost_breakdown.values()) if plan.cost_breakdown else 0,
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
            print(f"✓ Mitigation plans CSV: {plans_file}")
            print(f"✓ Individual plans JSON: {plans_dir}")
        else:
            print("✗ No mitigation plans generated")
        
        mitigation_time = time.time() - start_time
        
        # ================================================================
        # STEP 6: COMPREHENSIVE DATASET EXPORT
        # ================================================================
        
        print_section("Step 6: Dataset Export")
        
        # Create master Excel file with all datasets
        excel_file = output_dir / f"mitigation_demo_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Grid summary
            grid_summary = pd.DataFrame([{
                'total_buses': len(base_net.bus),
                'total_lines': len(base_net.line),
                'total_generators': len(base_net.gen),
                'base_generation_mw': base_result.total_generation_mw,
                'base_load_mw': base_result.total_load_mw,
                'base_violations': len(base_result.voltage_violations) + len(base_result.thermal_violations),
                'stressed_violations': len(stressed_result.voltage_violations) + len(stressed_result.thermal_violations)
            }])
            grid_summary.to_excel(writer, sheet_name='Grid_Summary', index=False)
            
            # Contingency analysis
            analysis_df.to_excel(writer, sheet_name='Contingency_Analysis', index=False)
            
            # Mitigation plans
            if mitigation_plans:
                plans_df.to_excel(writer, sheet_name='Mitigation_Plans', index=False)
        
        print(f"✓ Comprehensive Excel report: {excel_file}")
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        
        print_section("MITIGATION DEMO COMPLETED SUCCESSFULLY!")
        
        total_time = grid_time + scenario_time + contingency_gen_time + analysis_time + mitigation_time
        
        print(f"📊 Execution Summary:")
        print(f"  • Total time: {total_time:.2f} seconds")
        print(f"  • Grid generation: {grid_time:.2f}s")
        print(f"  • Scenario generation: {scenario_time:.2f}s") 
        print(f"  • Contingency generation: {contingency_gen_time:.2f}s")
        print(f"  • Contingency analysis: {analysis_time:.2f}s")
        print(f"  • Mitigation planning: {mitigation_time:.2f}s")
        
        print(f"\n📁 Generated Datasets:")
        print(f"  • Contingency analysis: {contingency_file}")
        if mitigation_plans:
            print(f"  • Mitigation plans CSV: {plans_file}")
            print(f"  • Individual plans JSON: {plans_dir}")
        print(f"  • Comprehensive Excel: {excel_file}")
        
        print(f"\n🎯 Key Results:")
        print(f"  • Grid: {len(base_net.bus)} buses, {len(base_net.line)} lines")
        print(f"  • Stress scenario: {len(stressed_result.voltage_violations) + len(stressed_result.thermal_violations)} violations")
        print(f"  • Contingencies: {len(contingency_results)} analyzed ({converged_contingencies} converged)")
        print(f"  • Mitigation: {len(mitigation_plans)} plans generated and executed")
        
        print(f"\n✅ All datasets including CSV mitigation plans available in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Mitigation demo completed successfully!")
        print("Check the outputs/mitigation_demo/ directory for CSV files with mitigation action plans.")
    else:
        print("\n💥 Demo failed. Check error messages above.")
        exit(1) 