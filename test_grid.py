#!/usr/bin/env python3
"""
Test script for AdvancedGrid creation
"""

import time
from grid import AdvancedGrid
from loguru import logger

def test_grid_creation():
    """Test creating the full 2000+ bus grid"""
    logger.info("Starting grid creation test...")
    
    start_time = time.time()
    
    # Create 10 regions with 200 buses each = 2000 buses
    regions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # Create the grid
    grid = AdvancedGrid(regions, buses_per_region=200)
    
    creation_time = time.time() - start_time
    logger.info(f"Grid creation completed in {creation_time:.2f} seconds")
    
    # Print summary
    summary = grid.get_summary()
    logger.info("Grid Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Test JSON export
    logger.info("Testing JSON export...")
    json_str = grid.to_json()
    logger.info(f"JSON export completed, size: {len(json_str)} characters")
    
    # Save to file
    with open("data/topology/test_grid.json", "w") as f:
        f.write(json_str)
    logger.info("Grid saved to data/topology/test_grid.json")
    
    return grid

if __name__ == "__main__":
    test_grid_creation() 