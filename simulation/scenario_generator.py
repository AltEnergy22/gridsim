"""
Realistic Power System Scenario Generator

Generates diverse operational scenarios including:
- Normal operations with natural fluctuations
- Stress cases near operational limits
- Challenging/failure scenarios for ML training
"""

import numpy as np
import pandas as pd
import random
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

from loguru import logger

class ScenarioType(Enum):
    """Types of scenarios to generate"""
    NORMAL = "normal"           # 80-85% - Normal operations with fluctuations
    STRESS = "stress"           # 10-15% - Near operational limits
    CHALLENGE = "challenge"     # 5% - Difficult/failure scenarios

@dataclass
class ScenarioConditions:
    """Operational conditions for a scenario"""
    timestamp: datetime.datetime
    scenario_type: ScenarioType
    load_multiplier: float      # Global load scaling factor
    renewable_cf: Dict[str, float]  # Capacity factors by renewable type
    fuel_costs: Dict[str, float]    # Fuel costs by type
    weather_temp_c: float
    economic_index: float
    voltage_stress_factor: float    # For creating voltage stress
    
    # Labeling for ML
    expected_convergence: bool
    stress_level: str  # "low", "medium", "high"
    
class ScenarioGenerator:
    """Generate realistic power system operational scenarios"""
    
    def __init__(self):
        # Scenario distribution weights
        self.scenario_weights = {
            ScenarioType.NORMAL: 0.80,    # 80% normal operations
            ScenarioType.STRESS: 0.15,    # 15% stress cases  
            ScenarioType.CHALLENGE: 0.05  # 5% challenging scenarios
        }
        
        # Seasonal and diurnal patterns
        self.seasonal_factors = {
            1: 1.15, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.85,
            7: 0.80, 8: 0.85, 9: 0.90, 10: 0.95, 11: 1.05, 12: 1.10
        }
        
        self.diurnal_pattern = [
            0.6, 0.55, 0.52, 0.50, 0.52, 0.58,  # 0-5: Night
            0.70, 0.85, 0.95, 1.00, 1.02, 1.05, # 6-11: Morning ramp
            1.08, 1.10, 1.05, 1.00, 1.02, 1.08, # 12-17: Afternoon peak
            1.15, 1.20, 1.10, 0.95, 0.80, 0.70  # 18-23: Evening peak
        ]
        
    def generate_scenario(self, target_type: Optional[ScenarioType] = None) -> ScenarioConditions:
        """Generate a single realistic scenario"""
        
        # Determine scenario type
        if target_type:
            scenario_type = target_type
        else:
            scenario_type = random.choices(
                list(self.scenario_weights.keys()),
                weights=list(self.scenario_weights.values())
            )[0]
        
        # Generate timestamp (random time in recent year)
        base_time = datetime.datetime.now()
        days_back = random.randint(0, 365)
        hour = random.randint(0, 23)
        timestamp = base_time - datetime.timedelta(days=days_back)
        timestamp = timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Generate conditions based on scenario type
        if scenario_type == ScenarioType.NORMAL:
            return self._generate_normal_scenario(timestamp)
        elif scenario_type == ScenarioType.STRESS:
            return self._generate_stress_scenario(timestamp)
        else:  # CHALLENGE
            return self._generate_challenge_scenario(timestamp)
            
    def _generate_normal_scenario(self, timestamp: datetime.datetime) -> ScenarioConditions:
        """Generate normal operational scenario with natural fluctuations"""
        
        # Base load multiplier from diurnal/seasonal patterns
        seasonal = self.seasonal_factors[timestamp.month]
        diurnal = self.diurnal_pattern[timestamp.hour]
        base_multiplier = seasonal * diurnal
        
        # Add random walk fluctuations ±5-10%
        load_noise = random.gauss(0, 0.07)  # 7% std dev
        load_multiplier = base_multiplier * (1 + load_noise)
        load_multiplier = max(0.3, min(1.5, load_multiplier))  # Reasonable bounds
        
        # Renewable capacity factors with realistic intermittency
        renewable_cf = self._generate_renewable_profiles(timestamp, intermittent=True)
        
        # Fuel costs with market fluctuations
        fuel_costs = self._generate_fuel_costs(market_stress=False)
        
        # Weather conditions
        weather_temp = self._generate_weather(timestamp, extreme=False)
        
        # Economic conditions
        economic_index = random.gauss(1.0, 0.05)  # ±5% economic variation
        
        return ScenarioConditions(
            timestamp=timestamp,
            scenario_type=ScenarioType.NORMAL,
            load_multiplier=load_multiplier,
            renewable_cf=renewable_cf,
            fuel_costs=fuel_costs,
            weather_temp_c=weather_temp,
            economic_index=economic_index,
            voltage_stress_factor=1.0,  # No voltage stress
            expected_convergence=True,
            stress_level="low"
        )
    
    def _generate_stress_scenario(self, timestamp: datetime.datetime) -> ScenarioConditions:
        """Generate stress scenario near operational limits"""
        
        # High load conditions (90-100% of system capacity)
        base_multiplier = 1.1 + random.uniform(0, 0.15)  # 110-125% of normal
        
        # Reduced renewable output during high load
        renewable_cf = self._generate_renewable_profiles(timestamp, intermittent=True, reduced=True)
        
        # Elevated fuel costs (market stress)
        fuel_costs = self._generate_fuel_costs(market_stress=True)
        
        # Extreme weather
        weather_temp = self._generate_weather(timestamp, extreme=True)
        
        # Economic stress
        economic_index = random.gauss(1.15, 0.1)  # 15% above normal
        
        # Voltage stress (tight reactive margins)
        voltage_stress = random.uniform(1.01, 1.02)  # Slightly higher voltages
        
        return ScenarioConditions(
            timestamp=timestamp,
            scenario_type=ScenarioType.STRESS,
            load_multiplier=base_multiplier,
            renewable_cf=renewable_cf,
            fuel_costs=fuel_costs,
            weather_temp_c=weather_temp,
            economic_index=economic_index,
            voltage_stress_factor=voltage_stress,
            expected_convergence=True,  # Should still converge but be challenging
            stress_level="high"
        )
    
    def _generate_challenge_scenario(self, timestamp: datetime.datetime) -> ScenarioConditions:
        """Generate challenging scenario that may not converge"""
        
        # Extreme load conditions
        load_multiplier = random.uniform(1.2, 1.4)  # 120-140% of normal
        
        # Poor renewable conditions
        renewable_cf = {fuel: cf * 0.3 for fuel, cf in 
                       self._generate_renewable_profiles(timestamp).items()}
        
        # Crisis fuel costs
        fuel_costs = self._generate_fuel_costs(market_stress=True)
        for fuel in fuel_costs:
            fuel_costs[fuel] *= random.uniform(1.5, 2.0)  # 50-100% price spike
        
        # Extreme weather
        weather_temp = self._generate_weather(timestamp, extreme=True)
        if random.random() < 0.5:
            weather_temp += random.uniform(5, 15)  # Heat wave
        
        # Economic crisis
        economic_index = random.gauss(1.3, 0.15)  # 30% above normal
        
        # Extreme voltage stress
        voltage_stress = random.uniform(1.03, 1.05)  # Very high voltage setpoints
        
        return ScenarioConditions(
            timestamp=timestamp,
            scenario_type=ScenarioType.CHALLENGE,
            load_multiplier=load_multiplier,
            renewable_cf=renewable_cf,
            fuel_costs=fuel_costs,
            weather_temp_c=weather_temp,
            economic_index=economic_index,
            voltage_stress_factor=voltage_stress,
            expected_convergence=False,  # May not converge
            stress_level="extreme"
        )
    
    def _generate_renewable_profiles(self, timestamp: datetime.datetime, 
                                   intermittent: bool = True,
                                   reduced: bool = False) -> Dict[str, float]:
        """Generate realistic renewable capacity factors"""
        
        # Base profiles
        hour = timestamp.hour
        month = timestamp.month
        
        # Solar profile (daylight hours with seasonal variation)
        if 6 <= hour <= 18:
            solar_base = math.sin(math.pi * (hour - 6) / 12)
            seasonal_solar = 0.7 + 0.3 * math.cos(2 * math.pi * (month - 6) / 12)
            solar_cf = solar_base * seasonal_solar
        else:
            solar_cf = 0.0
            
        # Wind profile (more variable, some seasonal pattern)
        wind_base = 0.3 + 0.2 * math.sin(2 * math.pi * hour / 24 + 3)
        seasonal_wind = 0.8 + 0.2 * math.cos(2 * math.pi * (month - 12) / 12)
        wind_cf = wind_base * seasonal_wind
        
        # Add intermittency (realistic hour-to-hour jumps)
        if intermittent:
            solar_cf *= random.gauss(1.0, 0.15)  # ±15% variation
            wind_cf *= random.gauss(1.0, 0.25)   # ±25% variation (more variable)
        
        # Reduce for stress scenarios
        if reduced:
            solar_cf *= random.uniform(0.5, 0.8)
            wind_cf *= random.uniform(0.6, 0.9)
            
        # Clamp to realistic bounds
        solar_cf = max(0.0, min(1.0, solar_cf))
        wind_cf = max(0.0, min(1.0, wind_cf))
        
        return {
            'solar': solar_cf,
            'wind': wind_cf,
            'hydro': random.uniform(0.4, 0.8)  # Seasonal hydro availability
        }
    
    def _generate_fuel_costs(self, market_stress: bool = False) -> Dict[str, float]:
        """Generate fuel costs with market fluctuations"""
        
        base_costs = {
            'gas': 4.0,      # $/MMBtu
            'coal': 2.5,     # $/MMBtu  
            'nuclear': 0.5,  # $/MMBtu (fuel only)
            'oil': 8.0       # $/MMBtu
        }
        
        # Add market fluctuations
        for fuel in base_costs:
            if market_stress:
                # Higher volatility during stress
                multiplier = random.gauss(1.2, 0.3)  # 20% higher with more volatility
            else:
                # Normal market fluctuations
                multiplier = random.gauss(1.0, 0.15)  # ±15% normal variation
                
            base_costs[fuel] *= max(0.5, multiplier)  # Don't go below 50% of base
            
        return base_costs
    
    def _generate_weather(self, timestamp: datetime.datetime, extreme: bool = False) -> float:
        """Generate weather temperature"""
        
        # Seasonal base temperature
        month = timestamp.month
        seasonal_temp = 15 + 10 * math.cos(2 * math.pi * (month - 7) / 12)  # Peak in July
        
        # Diurnal variation
        hour = timestamp.hour
        diurnal_temp = 5 * math.cos(2 * math.pi * (hour - 14) / 24)  # Peak at 2 PM
        
        base_temp = seasonal_temp + diurnal_temp
        
        if extreme:
            # Heat wave or cold snap
            if random.random() < 0.5:
                base_temp += random.uniform(8, 20)  # Heat wave
            else:
                base_temp -= random.uniform(5, 15)  # Cold snap
        else:
            # Normal weather variation
            base_temp += random.gauss(0, 3)  # ±3°C normal variation
            
        return base_temp
    
    def generate_scenario_batch(self, num_scenarios: int) -> List[ScenarioConditions]:
        """Generate a batch of diverse scenarios"""
        
        scenarios = []
        for _ in range(num_scenarios):
            scenario = self.generate_scenario()
            scenarios.append(scenario)
            
        # Log scenario distribution
        scenario_counts = {}
        for scenario in scenarios:
            scenario_counts[scenario.scenario_type] = scenario_counts.get(scenario.scenario_type, 0) + 1
            
        logger.info(f"Generated {num_scenarios} scenarios:")
        for stype, count in scenario_counts.items():
            logger.info(f"  {stype.value}: {count} ({count/num_scenarios*100:.1f}%)")
            
        return scenarios