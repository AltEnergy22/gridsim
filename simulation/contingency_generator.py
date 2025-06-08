"""
Contingency Generation Engine

Generates comprehensive N-1 and N-2 contingency scenarios for power system analysis.
Implements probability-based scenario enumeration with severity distributions.
"""

import json
import math
import random
import string
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from grid.advanced_grid import AdvancedGrid
from grid.grid_components import Contingency


@dataclass
class ContingencyMetadata:
    """Metadata for contingency scenarios"""
    total_scenarios: int
    n1_scenarios: int
    n2_scenarios: int
    topology_hash: str
    generation_timestamp: str
    failure_rates: Dict[str, float]
    
    
@dataclass 
class ScenarioFilter:
    """Filters for scenario generation"""
    include_n1: bool = True
    include_n2: bool = True
    max_n2_scenarios: int = 5000
    min_probability: float = 1e-8
    element_types: List[str] = field(default_factory=lambda: ['line', 'generator', 'transformer'])
    regions: Optional[List[str]] = None


class ContingencyGenerator:
    """
    Comprehensive contingency scenario generator
    
    Generates N-1 and selective N-2 scenarios with realistic probabilities
    and severity distributions based on equipment characteristics.
    """
    
    def __init__(self, grid: AdvancedGrid):
        self.grid = grid
        
        # Base failure rates (per year)
        self.base_failure_rates = {
            'line': 0.005,      # Base rate per km per year
            'generator': {
                'coal': 0.05,
                'gas': 0.07, 
                'nuclear': 0.03,
                'hydro': 0.03,
                'wind': 0.10,
                'solar': 0.10
            },
            'transformer': 0.01,  # Base rate per year
        }
        
        # Severity distribution parameters
        self.severity_params = {
            'line': {'distribution': 'lognormal', 'mean_mw': 50, 'sigma': 0.3},
            'generator': {'distribution': 'lognormal', 'mean_mw': 100, 'sigma': 0.4},
            'transformer': {'distribution': 'lognormal', 'mean_mw': 75, 'sigma': 0.35}
        }
        
    def generate_all_scenarios(self, 
                             output_dir: str = "data/contingencies",
                             filters: Optional[ScenarioFilter] = None) -> ContingencyMetadata:
        """
        Generate complete contingency scenario set
        
        Args:
            output_dir: Directory to save scenario files
            filters: Scenario generation filters
            
        Returns:
            Metadata about generated scenarios
        """
        
        if filters is None:
            filters = ScenarioFilter()
            
        logger.info("Starting comprehensive contingency generation...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scenarios = []
        
        # Generate N-1 scenarios
        if filters.include_n1:
            logger.info("Generating N-1 scenarios...")
            n1_scenarios = self._generate_n1_scenarios(filters)
            scenarios.extend(n1_scenarios)
            logger.info(f"Generated {len(n1_scenarios)} N-1 scenarios")
        
        # Generate N-2 scenarios
        if filters.include_n2:
            logger.info("Generating N-2 scenarios...")
            n2_scenarios = self._generate_n2_scenarios(filters)
            scenarios.extend(n2_scenarios)
            logger.info(f"Generated {len(n2_scenarios)} N-2 scenarios")
        
        # Calculate probabilities
        self._calculate_scenario_probabilities(scenarios)
        
        # Filter by minimum probability
        scenarios = [s for s in scenarios if s.probability >= filters.min_probability]
        
        # Sort by probability (highest first)
        scenarios.sort(key=lambda x: x.probability, reverse=True)
        
        # Save scenarios to files
        self._save_scenarios(scenarios, output_path)
        
        # Generate metadata
        metadata = ContingencyMetadata(
            total_scenarios=len(scenarios),
            n1_scenarios=len([s for s in scenarios if s.type == "N-1"]),
            n2_scenarios=len([s for s in scenarios if s.type == "N-2"]),
            topology_hash=self._calculate_topology_hash(),
            generation_timestamp=datetime.now().isoformat(),
            failure_rates=self._get_element_failure_rates()
        )
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)
        
        logger.info(f"Generated {len(scenarios)} total contingency scenarios")
        return metadata
    
    def _generate_n1_scenarios(self, filters: ScenarioFilter) -> List[Contingency]:
        """Generate N-1 (single element) scenarios"""
        scenarios = []
        
        # Line outages
        if 'line' in filters.element_types:
            for line_id, line in self.grid.lines.items():
                if filters.regions and not any(line_id.startswith(r) for r in filters.regions):
                    continue
                    
                scenario = Contingency(
                    id=f"N1_LINE_{line_id}",
                    name=f"Line Outage: {line.name}",
                    type="N-1",
                    elements=[line_id],
                    probability=0.0,  # Will be calculated later
                    severity_dist=self.severity_params['line'].copy(),
                    duration_min=random.uniform(30, 180)
                )
                scenarios.append(scenario)
        
        # Generator trips
        if 'generator' in filters.element_types:
            for gen_id, gen in self.grid.generators.items():
                if filters.regions and not any(gen.bus_id.startswith(r) for r in filters.regions):
                    continue
                    
                # Only include online generators
                if hasattr(gen, 'online') and not gen.online:
                    continue
                    
                scenario = Contingency(
                    id=f"N1_GEN_{gen_id}",
                    name=f"Generator Trip: {gen.name}",
                    type="N-1", 
                    elements=[gen_id],
                    probability=0.0,
                    severity_dist=self.severity_params['generator'].copy(),
                    duration_min=random.uniform(60, 300)
                )
                scenarios.append(scenario)
        
        # Transformer failures
        if 'transformer' in filters.element_types:
            for xfmr_id, xfmr in self.grid.transformers.items():
                scenario = Contingency(
                    id=f"N1_XFMR_{xfmr_id}",
                    name=f"Transformer Failure: {xfmr.name}",
                    type="N-1",
                    elements=[xfmr_id],
                    probability=0.0,
                    severity_dist=self.severity_params['transformer'].copy(),
                    duration_min=random.uniform(120, 480)
                )
                scenarios.append(scenario)
        
        return scenarios
    
    def _generate_n2_scenarios(self, filters: ScenarioFilter) -> List[Contingency]:
        """Generate selective N-2 (double element) scenarios"""
        scenarios = []
        
        # Strategy 1: Adjacent line pairs (same bus)
        scenarios.extend(self._generate_adjacent_line_pairs(filters))
        
        # Strategy 2: High-impact generator + critical line combinations
        scenarios.extend(self._generate_gen_line_combinations(filters))
        
        # Strategy 3: Transformer pairs at same substation
        scenarios.extend(self._generate_transformer_pairs(filters))
        
        # Strategy 4: Geographically correlated failures
        scenarios.extend(self._generate_geographic_pairs(filters))
        
        # Limit total N-2 scenarios
        if len(scenarios) > filters.max_n2_scenarios:
            # Sort by estimated impact and take top scenarios
            scenarios.sort(key=lambda s: self._estimate_scenario_impact(s), reverse=True)
            scenarios = scenarios[:filters.max_n2_scenarios]
        
        return scenarios
    
    def _generate_adjacent_line_pairs(self, filters: ScenarioFilter) -> List[Contingency]:
        """Generate N-2 scenarios for lines sharing common buses"""
        scenarios = []
        
        # Group lines by buses
        bus_to_lines = {}
        for line_id, line in self.grid.lines.items():
            for bus_id in [line.from_bus, line.to_bus]:
                if bus_id not in bus_to_lines:
                    bus_to_lines[bus_id] = []
                bus_to_lines[bus_id].append(line_id)
        
        # Generate pairs for buses with multiple lines
        for bus_id, lines in bus_to_lines.items():
            if len(lines) >= 2:
                for line_a, line_b in combinations(lines, 2):
                    scenario = Contingency(
                        id=f"N2_LINES_{line_a}_{line_b}",
                        name=f"Adjacent Lines: {line_a}, {line_b}",
                        type="N-2",
                        elements=[line_a, line_b],
                        probability=0.0,
                        severity_dist=self.severity_params['line'].copy(),
                        duration_min=random.uniform(60, 360)
                    )
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_gen_line_combinations(self, filters: ScenarioFilter) -> List[Contingency]:
        """Generate N-2 scenarios combining large generators with critical lines"""
        scenarios = []
        
        # Top capacity generators
        top_gens = sorted(self.grid.generators.items(), 
                         key=lambda x: x[1].capacity_mw, reverse=True)[:5]
        
        # Critical tie lines (interregional or high capacity)
        critical_lines = []
        for line_id, line in self.grid.lines.items():
            if (hasattr(line, 'transfer_limit_mw') and line.transfer_limit_mw and 
                line.transfer_limit_mw > 1000):
                critical_lines.append(line_id)
        
        # Take top critical lines
        critical_lines = critical_lines[:5]
        
        # Generate combinations
        for gen_id, gen in top_gens:
            for line_id in critical_lines:
                scenario = Contingency(
                    id=f"N2_GEN_{gen_id}_LINE_{line_id}",
                    name=f"Generator + Line: {gen.name}, {line_id}",
                    type="N-2",
                    elements=[gen_id, line_id],
                    probability=0.0,
                    severity_dist={'distribution': 'lognormal', 'mean_mw': 150, 'sigma': 0.5},
                    duration_min=random.uniform(90, 420)
                )
                scenarios.append(scenario)
        
        return scenarios
    
    def _generate_transformer_pairs(self, filters: ScenarioFilter) -> List[Contingency]:
        """Generate N-2 scenarios for transformer pairs at same substation"""
        scenarios = []
        
        # Group transformers by substation
        sub_to_xfmrs = {}
        for xfmr_id, xfmr in self.grid.transformers.items():
            # Extract substation from transformer ID (simplified)
            sub_id = xfmr_id.split('_')[1] if '_' in xfmr_id else 'UNKNOWN'
            if sub_id not in sub_to_xfmrs:
                sub_to_xfmrs[sub_id] = []
            sub_to_xfmrs[sub_id].append(xfmr_id)
        
        # Generate pairs for substations with multiple transformers
        for sub_id, xfmrs in sub_to_xfmrs.items():
            if len(xfmrs) >= 2:
                for xfmr_a, xfmr_b in combinations(xfmrs, 2):
                    scenario = Contingency(
                        id=f"N2_XFMR_{xfmr_a}_{xfmr_b}",
                        name=f"Transformer Pair: {xfmr_a}, {xfmr_b}",
                        type="N-2",
                        elements=[xfmr_a, xfmr_b],
                        probability=0.0,
                        severity_dist=self.severity_params['transformer'].copy(),
                        duration_min=random.uniform(180, 720)
                    )
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_geographic_pairs(self, filters: ScenarioFilter) -> List[Contingency]:
        """Generate N-2 scenarios for geographically correlated failures"""
        scenarios = []
        
        # Group elements by region
        for region in self.grid.regions:
            region_lines = [lid for lid in self.grid.lines.keys() if lid.startswith(f"L_{region}")]
            region_gens = [gid for gid, gen in self.grid.generators.items() 
                          if gen.bus_id.startswith(region)]
            
            # Sample some regional pairs
            if len(region_lines) >= 2:
                # Take up to 3 line pairs per region
                line_pairs = list(combinations(region_lines, 2))
                sampled_pairs = random.sample(line_pairs, min(3, len(line_pairs)))
                
                for line_a, line_b in sampled_pairs:
                    scenario = Contingency(
                        id=f"N2_REGIONAL_{region}_{line_a}_{line_b}",
                        name=f"Regional Outage: {region} - {line_a}, {line_b}",
                        type="N-2",
                        elements=[line_a, line_b],
                        probability=0.0,
                        severity_dist={'distribution': 'lognormal', 'mean_mw': 80, 'sigma': 0.4},
                        duration_min=random.uniform(120, 480)
                    )
                    scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_scenario_probabilities(self, scenarios: List[Contingency]):
        """Calculate normalized probabilities for all scenarios"""
        
        # Calculate individual element failure rates
        element_rates = {}
        
        # Line failure rates (age and length dependent)
        for line_id, line in self.grid.lines.items():
            age_years = getattr(line, 'age_years', random.randint(5, 30))
            length_km = getattr(line, 'length_km', 50.0)
            rate = self.base_failure_rates['line'] * math.exp(age_years / 30) * length_km / 100
            element_rates[line_id] = rate
        
        # Generator failure rates (type dependent)
        for gen_id, gen in self.grid.generators.items():
            rate = self.base_failure_rates['generator'].get(gen.type, 0.05)
            element_rates[gen_id] = rate
        
        # Transformer failure rates (age dependent)
        for xfmr_id, xfmr in self.grid.transformers.items():
            age_years = getattr(xfmr, 'age_years', random.randint(10, 40))
            rate = self.base_failure_rates['transformer'] * (age_years / 20)
            element_rates[xfmr_id] = rate
        
        # Calculate scenario probabilities
        total_probability = 0.0
        
        for scenario in scenarios:
            if scenario.type == "N-1":
                # Single element probability
                element_id = scenario.elements[0]
                scenario.probability = element_rates.get(element_id, 1e-6)
            elif scenario.type == "N-2":
                # Joint probability (assuming independence)
                prob = 1.0
                for element_id in scenario.elements:
                    prob *= element_rates.get(element_id, 1e-6)
                scenario.probability = prob
            
            total_probability += scenario.probability
        
        # Normalize probabilities to sum to 1
        if total_probability > 0:
            for scenario in scenarios:
                scenario.probability /= total_probability
    
    def _estimate_scenario_impact(self, scenario: Contingency) -> float:
        """Estimate the impact of a scenario for prioritization"""
        impact = 0.0
        
        for element_id in scenario.elements:
            if element_id in self.grid.generators:
                gen = self.grid.generators[element_id]
                impact += gen.capacity_mw
            elif element_id in self.grid.lines:
                line = self.grid.lines[element_id]
                impact += getattr(line, 'capacity_mw', 200)
            elif element_id in self.grid.transformers:
                xfmr = self.grid.transformers[element_id]
                impact += getattr(xfmr, 'capacity_mva', 100)
        
        return impact
    
    def _save_scenarios(self, scenarios: List[Contingency], output_path: Path):
        """Save scenarios to individual JSON files"""
        
        # Create scenarios index
        scenarios_index = []
        
        for scenario in scenarios:
            # Save individual scenario file
            scenario_file = output_path / f"{scenario.id}.json"
            with open(scenario_file, 'w') as f:
                # Convert to dict for JSON serialization
                scenario_dict = {
                    'id': scenario.id,
                    'name': scenario.name,
                    'type': scenario.type,
                    'elements': scenario.elements,
                    'probability': scenario.probability,
                    'severity_dist': scenario.severity_dist,
                    'duration_min': scenario.duration_min
                }
                json.dump(scenario_dict, f, indent=2)
            
            # Add to index
            scenarios_index.append({
                'id': scenario.id,
                'type': scenario.type,
                'probability': scenario.probability,
                'file': f"{scenario.id}.json"
            })
        
        # Save scenarios index
        index_file = output_path / "scenarios_index.json"
        with open(index_file, 'w') as f:
            json.dump(scenarios_index, f, indent=2)
    
    def _calculate_topology_hash(self) -> str:
        """Calculate hash of grid topology for traceability"""
        # Simple hash based on grid structure
        elements = []
        elements.extend(sorted(self.grid.buses.keys()))
        elements.extend(sorted(self.grid.lines.keys()))
        elements.extend(sorted(self.grid.generators.keys()))
        
        hash_input = ''.join(elements)
        return str(hash(hash_input))
    
    def _get_element_failure_rates(self) -> Dict[str, float]:
        """Get summary of failure rates used"""
        return {
            'base_line_rate_per_km_per_year': self.base_failure_rates['line'],
            'generator_rates_per_year': self.base_failure_rates['generator'],
            'transformer_base_rate_per_year': self.base_failure_rates['transformer']
        }


def load_contingencies(contingency_dir: str) -> List[Contingency]:
    """Load contingency scenarios from directory"""
    contingency_path = Path(contingency_dir)
    
    if not contingency_path.exists():
        logger.warning(f"Contingency directory {contingency_dir} does not exist")
        return []
    
    # Load scenarios index
    index_file = contingency_path / "scenarios_index.json"
    if not index_file.exists():
        logger.warning(f"Scenarios index file not found: {index_file}")
        return []
    
    with open(index_file, 'r') as f:
        scenarios_index = json.load(f)
    
    # Load individual scenarios
    contingencies = []
    for scenario_info in scenarios_index:
        scenario_file = contingency_path / scenario_info['file']
        if scenario_file.exists():
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
                contingency = Contingency(**scenario_data)
                contingencies.append(contingency)
    
    logger.info(f"Loaded {len(contingencies)} contingency scenarios")
    return contingencies


if __name__ == "__main__":
    # Example usage
    from grid.advanced_grid import AdvancedGrid
    
    # Create test grid
    regions = ['A', 'B', 'C']
    grid = AdvancedGrid(regions, buses_per_region=100)
    
    # Generate contingencies
    generator = ContingencyGenerator(grid)
    metadata = generator.generate_all_scenarios()
    
    print(f"Generated {metadata.total_scenarios} scenarios")
    print(f"N-1: {metadata.n1_scenarios}, N-2: {metadata.n2_scenarios}") 