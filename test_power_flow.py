#!/usr/bin/env python3
"""
Test Power Flow with Advanced Convergence at Full Scale
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from grid.advanced_grid import AdvancedGrid
from simulation.build_network import build_pandapower_network
from simulation.power_flow import PowerFlowEngine
import time

def test_full_scale_power_flow():
    """Test power flow convergence at full 2000+ bus scale"""
    logger.info("="*60)
    logger.info("TESTING FULL SCALE POWER FLOW WITH ADVANCED CONVERGENCE")
    logger.info("="*60)
    
    try:
        # Create full-scale grid (2000+ buses)
        logger.info("Creating full-scale grid...")
        start_time = time.time()
        
        # Create 10 regions for 2000 bus system
        regions = ['North', 'South', 'East', 'West', 'Central', 
                   'Northeast', 'Northwest', 'Southeast', 'Southwest', 'Midwest']
        
        grid = AdvancedGrid(
            regions=regions, 
            buses_per_region=200  # Full 2000 bus system
        )
        
        creation_time = time.time() - start_time
        logger.info(f"Grid created in {creation_time:.2f}s")
        logger.info(f"Grid stats: {len(grid.buses)} buses, {len(grid.lines)} lines, "
                   f"{len(grid.generators)} generators")
        
        # Build pandapower network
        logger.info("Building pandapower network...")
        start_time = time.time()
        
        net, builder = build_pandapower_network(grid)
        
        build_time = time.time() - start_time
        logger.info(f"Network built in {build_time:.2f}s")
        logger.info(f"Pandapower network: {len(net.bus)} buses, {len(net.line)} lines, "
                   f"{len(net.gen)} generators")
        
        # Display generation/load balance before power flow
        total_gen_capacity = net.gen['max_p_mw'].sum()
        total_load = net.load['p_mw'].sum()
        total_gen_dispatch = net.gen['p_mw'].sum()
        
        logger.info(f"Generation capacity: {total_gen_capacity:.1f} MW")
        logger.info(f"Total load: {total_load:.1f} MW")
        logger.info(f"Initial dispatch: {total_gen_dispatch:.1f} MW")
        logger.info(f"Reserve margin: {((total_gen_capacity - total_load) / total_load * 100):.1f}%")
        
        # Run power flow with advanced convergence
        logger.info("Starting advanced power flow analysis...")
        engine = PowerFlowEngine()
        
        start_time = time.time()
        results = engine.solve_power_flow(net)
        solve_time = time.time() - start_time
        
        # Display results
        logger.info("="*60)
        logger.info("POWER FLOW RESULTS")
        logger.info("="*60)
        
        if results.converged:
            logger.success(f"✅ POWER FLOW CONVERGED!")
            logger.info(f"Algorithm used: {results.algorithm_used.value}")
            logger.info(f"Iterations: {results.iterations}")
            logger.info(f"Solve time: {solve_time:.3f}s")
            logger.info(f"Total generation: {results.total_generation_mw:.1f} MW")
            logger.info(f"Total load: {results.total_load_mw:.1f} MW")
            logger.info(f"Total losses: {results.total_losses_mw:.1f} MW ({results.total_losses_mw/results.total_load_mw*100:.2f}%)")
            logger.info(f"Voltage range: {results.min_voltage_pu:.3f} - {results.max_voltage_pu:.3f} pu")
            logger.info(f"Max line loading: {results.max_line_loading_pct:.1f}%")
            logger.info(f"Max transformer loading: {results.max_trafo_loading_pct:.1f}%")
            
            # Violations summary
            if results.voltage_violations:
                logger.warning(f"⚠️ {len(results.voltage_violations)} voltage violations detected")
                for violation in results.voltage_violations[:5]:  # Show first 5
                    logger.warning(f"  {violation['bus_name']}: {violation['voltage_pu']:.3f} pu ({violation['violation_type']})")
            else:
                logger.success("✅ No voltage violations")
                
            if results.thermal_violations:
                logger.warning(f"⚠️ {len(results.thermal_violations)} thermal violations detected")
                for violation in results.thermal_violations[:5]:  # Show first 5
                    logger.warning(f"  {violation['element_name']}: {violation['loading_percent']:.1f}% loading")
            else:
                logger.success("✅ No thermal violations")
                
            # Export results
            try:
                output_file = "outputs/full_scale_power_flow_results.xlsx"
                os.makedirs("outputs", exist_ok=True)
                engine.export_results(results, output_file)
                logger.info(f"Results exported to {output_file}")
            except Exception as e:
                logger.warning(f"Could not export results: {e}")
                
        else:
            logger.error(f"❌ POWER FLOW FAILED TO CONVERGE")
            logger.error(f"Status: {results.status.value}")
            logger.error(f"Last algorithm tried: {results.algorithm_used.value}")
            logger.error(f"Solve time: {solve_time:.3f}s")
            
            # Diagnostic information
            logger.info("Network diagnostic information:")
            logger.info(f"  Buses: {len(net.bus)}")
            logger.info(f"  Lines: {len(net.line)}")
            logger.info(f"  Transformers: {len(net.trafo)}")
            logger.info(f"  Generators: {len(net.gen)}")
            logger.info(f"  Loads: {len(net.load)}")
            
            # Check for obvious issues
            if len(net.ext_grid) == 0:
                logger.error("❌ No external grid connection found!")
            
            zero_impedance_lines = ((net.line['r_ohm_per_km'] == 0) | 
                                   (net.line['x_ohm_per_km'] == 0)).sum()
            if zero_impedance_lines > 0:
                logger.warning(f"⚠️ {zero_impedance_lines} lines with zero impedance")
                
        logger.info("="*60)
        return results.converged
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Testing Full Scale Power Flow with Advanced Convergence")
    success = test_full_scale_power_flow()
    
    if success:
        logger.success("🎉 Full scale power flow test PASSED!")
    else:
        logger.error("💥 Full scale power flow test FAILED!")
        sys.exit(1) 