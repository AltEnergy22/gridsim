"""
Blended Topology System Demo

Demonstrates the comprehensive grid simulation capabilities with a manageable test case.
Shows integration of all components: topology, contingencies, mitigation, and ML.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# Configure logging for demo
logger.remove()
logger.add(lambda msg: print(msg, end=''), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

# Core components
from grid.advanced_grid import AdvancedGrid
from simulation.build_network import build_pandapower_network
from simulation.power_flow import PowerFlowEngine
from simulation.contingency_generator import ContingencyGenerator, ScenarioFilter
from simulation.contingency_analyzer import ContingencyAnalyzer
from simulation.mitigation_engine import MitigationEngine

def print_banner(title: str):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_section(title: str):
    """Print a section header"""
    print(f"\n📋 {title}")
    print("-" * 60)

def demo_blended_topology():
    """
    Comprehensive demonstration of the blended topology system
    
    Shows all major capabilities:
    1. Advanced grid topology generation
    2. Contingency scenario generation
    3. Contingency analysis with violation detection
    4. Mitigation planning and execution
    5. Results export and analysis
    """
    
    print_banner("BLENDED TOPOLOGY GRID SIMULATION SYSTEM DEMO")
    
    # Demo configuration
    demo_config = {
        'regions': ['A', 'B', 'C'],  # Smaller grid for demo
        'buses_per_region': 100,     # 300 total buses
        'max_contingencies': 50,     # Limit for demo
        'demo_mode': True
    }
    
    print(f"Demo Configuration: {json.dumps(demo_config, indent=2)}")
    
    # Create output directory
    demo_output = Path("outputs/demo")
    demo_output.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # ============================================================================
        # PHASE 1: ADVANCED GRID TOPOLOGY GENERATION
        # ============================================================================
        
        print_section("Phase 1: Advanced Grid Topology Generation")
        
        start_time = time.time()
        
        # Create comprehensive grid with all advanced features
        logger.info("Creating advanced grid topology with comprehensive modeling...")
        
        grid = AdvancedGrid(
            regions=demo_config['regions'],
            buses_per_region=demo_config['buses_per_region']
        )
        
        # Display grid statistics
        summary = grid.get_summary()
        print(f"✓ Grid created successfully:")
        print(f"  • Buses: {summary['total_buses']}")
        print(f"  • Generators: {summary['total_generators']} ({summary['total_capacity_mw']:.1f} MW capacity)")
        print(f"  • Lines: {summary['total_lines']}")
        print(f"  • Substations: {summary['total_substations']}")
        print(f"  • Voltage levels: {summary['voltage_levels']}")
        
        # Build pandapower network
        logger.info("Building pandapower network for power flow analysis...")
        base_net, network_builder = build_pandapower_network(grid)
        
        print(f"✓ Pandapower network built: {len(base_net.bus)} buses, {len(base_net.line)} lines")
        
        # Verify base case power flow
        logger.info("Running base case power flow...")
        pf_engine = PowerFlowEngine()
        pf_result = pf_engine.solve_power_flow(base_net)
        
        if pf_result.converged:
            print(f"✓ Base case power flow converged in {pf_result.solve_time_sec:.3f}s")
            print(f"  • Iterations: {pf_result.iterations}")
            print(f"  • Total generation: {pf_result.total_generation_mw:.1f} MW")
            print(f"  • Total load: {pf_result.total_load_mw:.1f} MW")
            print(f"  • Total losses: {pf_result.total_losses_mw:.1f} MW ({pf_result.total_losses_mw/pf_result.total_load_mw*100:.2f}%)")
            print(f"  • Voltage range: {pf_result.min_voltage_pu:.3f} - {pf_result.max_voltage_pu:.3f} pu")
            print(f"  • Max line loading: {pf_result.max_line_loading_pct:.1f}%")
            
            if pf_result.voltage_violations:
                print(f"  ⚠️ Voltage violations: {len(pf_result.voltage_violations)}")
            if pf_result.thermal_violations:
                print(f"  ⚠️ Thermal violations: {len(pf_result.thermal_violations)}")
        else:
            print("❌ Base case power flow did not converge!")
            return
        
        topology_time = time.time() - start_time
        results['topology'] = {
            'grid_summary': summary,
            'base_case_converged': pf_result.converged,
            'generation_time': topology_time
        }
        
        # ============================================================================
        # PHASE 2: CONTINGENCY SCENARIO GENERATION
        # ============================================================================
        
        print_section("Phase 2: Contingency Scenario Generation")
        
        start_time = time.time()
        
        # Create scenario filter for demo
        scenario_filter = ScenarioFilter(
            include_n1=True,
            include_n2=True,
            max_n2_scenarios=demo_config['max_contingencies'],
            min_probability=1e-6,
            element_types=['line', 'generator', 'transformer']
        )
        
        # Generate contingency scenarios
        logger.info("Generating N-1 and N-2 contingency scenarios...")
        generator = ContingencyGenerator(grid)
        
        # Use demo output directory
        demo_contingency_dir = demo_output / "contingencies"
        metadata = generator.generate_all_scenarios(
            output_dir=str(demo_contingency_dir),
            filters=scenario_filter
        )
        
        print(f"✓ Contingency scenarios generated:")
        print(f"  • Total scenarios: {metadata.total_scenarios}")
        print(f"  • N-1 scenarios: {metadata.n1_scenarios}")
        print(f"  • N-2 scenarios: {metadata.n2_scenarios}")
        print(f"  • Topology hash: {metadata.topology_hash}")
        
        contingency_gen_time = time.time() - start_time
        results['contingency_generation'] = {
            'total_scenarios': metadata.total_scenarios,
            'generation_time': contingency_gen_time
        }
        
        # ============================================================================
        # PHASE 3: CONTINGENCY ANALYSIS
        # ============================================================================
        
        print_section("Phase 3: Comprehensive Contingency Analysis")
        
        start_time = time.time()
        
        # Load generated scenarios
        from simulation.contingency_generator import load_contingencies
        contingencies = load_contingencies(str(demo_contingency_dir))
        
        # Limit for demo
        demo_contingencies = contingencies[:demo_config['max_contingencies']]
        
        logger.info(f"Analyzing {len(demo_contingencies)} contingency scenarios...")
        
        # Run contingency analysis
        analyzer = ContingencyAnalyzer(grid)
        analysis_results = analyzer.analyze_all_contingencies(
            base_net, demo_contingencies, parallel=True, max_workers=4
        )
        
        # Analyze results
        converged_results = [r for r in analysis_results if r.converged]
        results_with_violations = [r for r in converged_results if r.new_violations]
        critical_results = [r for r in converged_results if r.criticality_level in ['critical', 'high']]
        
        print(f"✓ Contingency analysis completed:")
        print(f"  • Scenarios analyzed: {len(analysis_results)}")
        print(f"  • Converged: {len(converged_results)} ({len(converged_results)/len(analysis_results)*100:.1f}%)")
        print(f"  • With new violations: {len(results_with_violations)}")
        print(f"  • Critical/High severity: {len(critical_results)}")
        
        if converged_results:
            avg_solve_time = sum(r.solve_time for r in converged_results) / len(converged_results)
            total_violations = sum(len(r.new_violations) for r in converged_results)
            print(f"  • Average solve time: {avg_solve_time:.3f}s")
            print(f"  • Total new violations: {total_violations}")
        
        # Show some detailed examples
        if critical_results:
            print(f"\n🔍 Critical Contingency Examples:")
            for i, result in enumerate(critical_results[:3]):
                print(f"  {i+1}. {result.contingency.id} ({result.contingency.type})")
                print(f"     Severity: {result.criticality_level}, Score: {result.severity_score:.2f}")
                print(f"     New violations: {len(result.new_violations)}")
                if result.new_violations:
                    violation = result.new_violations[0]
                    print(f"     Example: {violation.violation_type} at {violation.element_id} (severity: {violation.severity:.3f})")
        
        analysis_time = time.time() - start_time
        results['contingency_analysis'] = {
            'scenarios_analyzed': len(analysis_results),
            'converged': len(converged_results),
            'with_violations': len(results_with_violations),
            'critical_scenarios': len(critical_results),
            'analysis_time': analysis_time
        }
        
        # ============================================================================
        # PHASE 4: MITIGATION PLANNING AND EXECUTION
        # ============================================================================
        
        print_section("Phase 4: Mitigation Planning and Execution")
        
        start_time = time.time()
        
        # Initialize mitigation engine
        logger.info("Generating mitigation plans for contingencies with violations...")
        mitigation_engine = MitigationEngine(grid)
        
        # Generate mitigation plans
        plans = []
        execution_results = []
        
        for i, result in enumerate(results_with_violations[:20]):  # Demo limit
            try:
                plan = mitigation_engine.generate_mitigation_plan(result, base_net)
                if plan:
                    plans.append(plan)
                    
                    # Execute plan for demonstration
                    execution_result = mitigation_engine.execute_mitigation_plan(plan, base_net)
                    execution_results.append(execution_result)
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"Generated {i + 1} mitigation plans...")
                        
            except Exception as e:
                logger.warning(f"Error generating plan for {result.contingency.id}: {e}")
        
        print(f"✓ Mitigation planning completed:")
        print(f"  • Plans generated: {len(plans)}")
        print(f"  • Plans executed: {len(execution_results)}")
        
        if execution_results:
            successful = sum(1 for r in execution_results if r.success)
            total_cost = sum(r.final_cost_usd for r in execution_results)
            avg_cost = total_cost / len(execution_results)
            violations_cleared = sum(r.violations_cleared for r in execution_results)
            
            print(f"  • Successful executions: {successful}/{len(execution_results)} ({successful/len(execution_results)*100:.1f}%)")
            print(f"  • Total mitigation cost: ${total_cost:,.2f}")
            print(f"  • Average cost per plan: ${avg_cost:,.2f}")
            print(f"  • Total violations cleared: {violations_cleared}")
            
            # Show example plans
            print(f"\n🛠️ Example Mitigation Plans:")
            for i, plan in enumerate(plans[:3]):
                print(f"  {i+1}. {plan.action_id}")
                print(f"     Contingency: {plan.contingency_id}")
                print(f"     Actions: {len(plan.action_sequence)}")
                if plan.action_sequence:
                    action = plan.action_sequence[0]
                    print(f"     First action: {action.type} on {action.target}")
                    if 'delta_mw' in action.parameters:
                        print(f"     Amount: {action.parameters['delta_mw']:.1f} MW")
                print(f"     Estimated cost: ${sum(plan.cost_breakdown.values()):,.2f}")
        
        mitigation_time = time.time() - start_time
        results['mitigation'] = {
            'plans_generated': len(plans),
            'plans_executed': len(execution_results),
            'successful_executions': sum(1 for r in execution_results if r.success),
            'generation_time': mitigation_time
        }
        
        # ============================================================================
        # PHASE 5: RESULTS EXPORT AND ANALYSIS
        # ============================================================================
        
        print_section("Phase 5: Results Export and Analysis")
        
        start_time = time.time()
        
        # Export comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Contingency analysis results
        analysis_df = analyzer.export_results_to_dataframe(analysis_results)
        analysis_file = demo_output / f"contingency_analysis_{timestamp}.csv"
        analysis_df.to_csv(analysis_file, index=False)
        print(f"✓ Contingency analysis exported: {analysis_file}")
        
        # 2. Mitigation plans summary
        if plans:
            plans_summary = []
            for plan in plans:
                plans_summary.append({
                    'action_id': plan.action_id,
                    'contingency_id': plan.contingency_id,
                    'num_actions': len(plan.action_sequence),
                    'primary_action_type': plan.action_sequence[0].type if plan.action_sequence else 'none',
                    'estimated_cost_usd': sum(plan.cost_breakdown.values()),
                    'estimated_time_s': plan.execution_profile.get('estimated_time_s', 0)
                })
            
            plans_df = pd.DataFrame(plans_summary)
            plans_file = demo_output / f"mitigation_plans_{timestamp}.csv"
            plans_df.to_csv(plans_file, index=False)
            print(f"✓ Mitigation plans exported: {plans_file}")
        
        # 3. Excel comprehensive report
        excel_file = demo_output / f"demo_comprehensive_report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Grid summary
            pd.DataFrame([summary]).to_excel(writer, sheet_name='Grid_Summary', index=False)
            
            # Contingency analysis
            analysis_df.to_excel(writer, sheet_name='Contingency_Analysis', index=False)
            
            # Critical contingencies
            if critical_results:
                critical_df = analyzer.export_results_to_dataframe(critical_results)
                critical_df.to_excel(writer, sheet_name='Critical_Contingencies', index=False)
            
            # Mitigation plans
            if plans:
                plans_df.to_excel(writer, sheet_name='Mitigation_Plans', index=False)
        
        print(f"✓ Comprehensive Excel report: {excel_file}")
        
        # 4. System summary
        system_summary = {
            'demo_info': {
                'timestamp': datetime.now().isoformat(),
                'configuration': demo_config,
                'total_execution_time': sum([
                    results['topology']['generation_time'],
                    results['contingency_generation']['generation_time'],
                    results['contingency_analysis']['analysis_time'],
                    results['mitigation']['generation_time']
                ])
            },
            'results': results
        }
        
        summary_file = demo_output / f"demo_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(system_summary, f, indent=2, default=str)
        
        print(f"✓ System summary: {summary_file}")
        
        export_time = time.time() - start_time
        results['export'] = {'export_time': export_time}
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        
        print_banner("DEMO EXECUTION COMPLETED SUCCESSFULLY!")
        
        total_time = system_summary['demo_info']['total_execution_time']
        
        print(f"📊 Overall Results:")
        print(f"  • Total execution time: {total_time:.2f} seconds")
        print(f"  • Grid size: {summary['total_buses']} buses, {summary['total_generators']} generators")
        print(f"  • Contingencies analyzed: {results['contingency_analysis']['scenarios_analyzed']}")
        print(f"  • Convergence rate: {results['contingency_analysis']['converged']/results['contingency_analysis']['scenarios_analyzed']*100:.1f}%")
        print(f"  • Critical scenarios: {results['contingency_analysis']['critical_scenarios']}")
        print(f"  • Mitigation plans: {results['mitigation']['plans_generated']}")
        print(f"  • Successful executions: {results['mitigation']['successful_executions']}")
        
        print(f"\n📁 Output files available in: {demo_output}")
        print(f"  • {analysis_file.name}")
        if plans:
            print(f"  • {plans_file.name}")
        print(f"  • {excel_file.name}")
        print(f"  • {summary_file.name}")
        
        print(f"\n🎯 Key Capabilities Demonstrated:")
        print(f"  ✓ Advanced grid topology generation with comprehensive modeling")
        print(f"  ✓ Systematic N-1 and N-2 contingency scenario generation")
        print(f"  ✓ High-performance contingency analysis with violation detection")
        print(f"  ✓ Rule-based mitigation planning with economic optimization")
        print(f"  ✓ Plan execution and effectiveness evaluation")
        print(f"  ✓ Comprehensive data export for further analysis")
        
        print(f"\n🚀 Next Steps:")
        print(f"  • Scale up to full 2000+ bus system")
        print(f"  • Integrate machine learning pipeline (RF → GNN → RL)")
        print(f"  • Add real-time operational scenarios")
        print(f"  • Implement advanced dynamic analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo execution failed: {e}")
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎪 Starting Blended Topology System Demonstration...")
    print("This demo showcases all major capabilities with a manageable test case.")
    
    success = demo_blended_topology()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("Check the outputs/demo/ directory for detailed results.")
        print("\nTo run the full system: python run_blended_topology_system.py --mode full")
    else:
        print("\n💥 Demo failed. Check logs for details.")
        exit(1) 