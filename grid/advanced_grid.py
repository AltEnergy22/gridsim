"""
Advanced Grid Topology - Comprehensive Implementation
Based on requirements for 2000+ bus system with full modeling capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses_json import dataclass_json
import random
import string
import datetime
import math
import json
import numpy as np
from loguru import logger

# ========================= Helpers =========================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in km"""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def generate_id(prefix: str = "ID") -> str:
    """Generate unique ID with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{prefix}_{timestamp}_{random_suffix}"

# ========================= Load Modeling =========================

@dataclass_json
@dataclass
class LoadProfile:
    """Detailed load profile with temporal and external dependencies"""
    base_mw: float
    diurnal: List[float]  # 24-hour profile [0-23]
    weekly: Dict[int, float]  # weekday factors {0-6}
    seasonal: Dict[int, float]  # monthly factors {1-12}
    temp_sensitivity: float  # MW/°C above 20°C
    econ_sensitivity: float  # MW per economic index unit
    dr_capacity_mw: float  # demand response capability
    
    def get_base_load(self, ts: datetime.datetime) -> float:
        """Calculate base load for given timestamp"""
        diurnal_factor = self.diurnal[ts.hour]
        weekly_factor = self.weekly[ts.weekday()]
        seasonal_factor = self.seasonal[ts.month]
        return self.base_mw * diurnal_factor * weekly_factor * seasonal_factor

@dataclass_json
@dataclass 
class CustomerLoad:
    """Aggregated customer load by class"""
    residential: LoadProfile
    commercial: LoadProfile
    industrial: LoadProfile
    special: Dict[str, float]  # special loads (hospitals, data centers, etc.)
    
    def total_load(self, ts: datetime.datetime, temp_c: float, 
                   econ_idx: float, dr_signal: bool) -> float:
        """Calculate total load considering all factors"""
        # Base load from all customer classes
        load = (
            self.residential.get_base_load(ts) +
            self.commercial.get_base_load(ts) + 
            self.industrial.get_base_load(ts)
        )
        
        # Temperature sensitivity (cooling load)
        temp_load = ((self.residential.temp_sensitivity + 
                     self.commercial.temp_sensitivity) * max(0, temp_c - 20))
        
        # Economic sensitivity (mainly industrial)
        econ_load = self.industrial.econ_sensitivity * (econ_idx - 1.0)
        
        # Special loads
        special_load = sum(self.special.values())
        
        total = load + temp_load + econ_load + special_load
        
        # Apply demand response if signaled
        if dr_signal:
            curtailable = (self.residential.dr_capacity_mw +
                          self.commercial.dr_capacity_mw +
                          self.industrial.dr_capacity_mw)
            total = max(0, total - curtailable)
            
        return total

# ========================= Generation =========================

@dataclass_json
@dataclass
class Generator:
    """Comprehensive generator model"""
    id: str
    name: str
    bus_id: str
    type: str  # fuel type: coal, gas, nuclear, hydro, wind, solar
    capacity_mw: float
    min_up_time_hr: float
    min_down_time_hr: float
    ramp_rate_mw_per_min: float
    startup_cost: float
    heat_rate: float  # BTU/kWh
    fuel_cost: float  # $/MBTU
    variable_om: float  # $/MWh
    cf_profile: Optional[List[float]] = None  # capacity factor profile for renewables
    reservoir_level: Optional[float] = None  # % for hydro
    is_baseload: bool = False
    is_peaking: bool = False
    online: bool = False
    up_time_hr: float = 0.0
    down_time_hr: float = 0.0
    
    def available_capacity(self, ts: datetime.datetime) -> float:
        """Calculate available capacity considering constraints"""
        if self.type in ('wind', 'solar') and self.cf_profile:
            return self.capacity_mw * self.cf_profile[ts.hour]
        
        if self.type == 'hydro' and self.reservoir_level is not None:
            # Seasonal hydro pattern
            seasonal_factor = 0.5 + 0.5 * math.sin((ts.month/12) * 2 * math.pi)
            return self.capacity_mw * min(1.0, self.reservoir_level/100 * seasonal_factor)
            
        return self.capacity_mw
    
    def marginal_cost(self) -> float:
        """Calculate marginal cost $/MWh"""
        return (self.heat_rate * self.fuel_cost / 1000) + self.variable_om

# ========================= Network Elements =========================

@dataclass_json
@dataclass
class Bus:
    """Enhanced bus with full load and generation modeling"""
    id: str
    name: str
    lat: float
    lon: float
    voltage_level: int  # kV
    island: Optional[str] = None
    load: Optional[CustomerLoad] = None
    generators: List[str] = field(default_factory=list)
    substation_id: Optional[str] = None
    zone: str = "ZONE_1"
    
    def __post_init__(self):
        if self.load is None:
            self.load = self._create_default_load()
    
    def _create_default_load(self) -> CustomerLoad:
        """Create default load profile"""
        diurnal = [0.6 + 0.4 * math.sin((h-6)/24 * 2 * math.pi) for h in range(24)]
        weekly = {i: (1.0 if i < 5 else 0.8) for i in range(7)}
        seasonal = {m: (1.0 + 0.2 * math.cos((m-1)/12 * 2 * math.pi)) for m in range(1, 13)}
        
        base_profile = LoadProfile(
            base_mw=random.uniform(1, 5),  # Reduced from 5-20 to 1-5 MW per bus
            diurnal=diurnal,
            weekly=weekly,
            seasonal=seasonal,
            temp_sensitivity=random.uniform(0.01, 0.05),  # Reduced temperature sensitivity
            econ_sensitivity=random.uniform(0.1, 0.5),    # Reduced economic sensitivity
            dr_capacity_mw=random.uniform(0, 1)           # Reduced DR capacity
        )
        
        special = {
            'hospital': random.uniform(0, 0.5),     # Reduced from 0-2 to 0-0.5
            'datacenter': random.uniform(0, 0.3),   # Reduced from 0-1 to 0-0.3
            'ev_charging': random.uniform(0, 0.5)   # Reduced from 0-2 to 0-0.5
        }
        
        return CustomerLoad(base_profile, base_profile, base_profile, special)

@dataclass_json
@dataclass
class Transformer:
    """Transformer with comprehensive modeling"""
    id: str
    name: str
    from_bus: str
    to_bus: str
    from_level: int  # kV
    to_level: int    # kV
    capacity_mva: float
    impedance_pu: float
    tap_ratio: float = 1.0
    tap_min: float = 0.9
    tap_max: float = 1.1
    tap_step: float = 0.0125  # 1.25% steps
    age_years: int = 0
    
    def __post_init__(self):
        if self.age_years == 0:
            self.age_years = random.randint(5, 40)

@dataclass_json
@dataclass
class EHVLine:
    """Comprehensive transmission line model"""
    id: str
    name: str
    from_bus: str
    to_bus: str
    voltage_kv: int
    capacity_mw: float
    length_km: float
    r_ohm_per_km: float
    x_ohm_per_km: float
    c_nf_per_km: float = 0.0
    terrain_factor: float = 1.0
    row_restriction: Optional[str] = None
    transfer_limit_mw: Optional[float] = None
    rating_normal: float = 0.0
    rating_short_term: float = 0.0
    rating_long_term: float = 0.0
    age_years: int = 0
    failure_rate: float = 0.005
    is_tie_line: bool = False
    
    def __post_init__(self):
        if self.rating_normal == 0.0:
            self.rating_normal = self.capacity_mw
            self.rating_short_term = self.capacity_mw * 1.2
            self.rating_long_term = self.capacity_mw * 1.1
        
        if self.age_years == 0:
            self.age_years = random.randint(0, 50)
            
        # Calculate failure rate based on age
        self.failure_rate = 0.005 * math.exp(self.age_years / 30)
    
    @classmethod
    def create_line(cls, from_bus: 'Bus', to_bus: 'Bus', voltage_kv: int, 
                   capacity_mw: float, **kwargs) -> 'EHVLine':
        """Create line with calculated distance"""
        distance = haversine(from_bus.lat, from_bus.lon, to_bus.lat, to_bus.lon)
        
        # Default electrical parameters by voltage level
        params = {
            765: {'r': 0.008, 'x': 0.025, 'c': 12.0},
            500: {'r': 0.010, 'x': 0.030, 'c': 10.0},
            345: {'r': 0.015, 'x': 0.040, 'c': 8.0},
            230: {'r': 0.020, 'x': 0.050, 'c': 6.0},
            138: {'r': 0.030, 'x': 0.070, 'c': 4.0},
            115: {'r': 0.035, 'x': 0.080, 'c': 3.5},
            69: {'r': 0.050, 'x': 0.100, 'c': 2.0}
        }
        
        level_params = params.get(voltage_kv, params[138])
        
        return cls(
            id=f"L_{from_bus.id}_{to_bus.id}",
            name=f"Line {from_bus.name}-{to_bus.name}",
            from_bus=from_bus.id,
            to_bus=to_bus.id,
            voltage_kv=voltage_kv,
            capacity_mw=capacity_mw,
            length_km=distance,
            r_ohm_per_km=level_params['r'],
            x_ohm_per_km=level_params['x'],
            c_nf_per_km=level_params['c'],
            **kwargs
        )

# ========================= Main AdvancedGrid Class =========================

class AdvancedGrid:
    """
    Comprehensive grid container with 2000+ buses
    Implements the full blended topology specification
    """
    
    def __init__(self, regions: List[str], buses_per_region: int = 200):
        from .grid_components import (
            Substation, Feeder, DER, Microgrid, ProtectionRelay, 
            PMU, AGCInterface, SmartMeter, FACTSDevice, EnergyStorage,
            Contingency, MaintenanceSchedule
        )
        
        self.regions = regions
        self.buses_per_region = buses_per_region
        
        # Core infrastructure
        self.buses: Dict[str, Bus] = {}
        self.generators: Dict[str, Generator] = {}
        self.transformers: Dict[str, Transformer] = {}
        self.lines: Dict[str, EHVLine] = {}
        
        # Grid infrastructure
        self.substations: Dict[str, Substation] = {}
        self.feeders: Dict[str, Feeder] = {}
        
        # Smart grid components
        self.smart_meters: Dict[str, SmartMeter] = {}
        self.ders: Dict[str, DER] = {}
        self.microgrids: Dict[str, Microgrid] = {}
        self.storage: Dict[str, EnergyStorage] = {}
        
        # Protection & control
        self.relays: Dict[str, ProtectionRelay] = {}
        self.pmus: Dict[str, PMU] = {}
        self.facts_devices: Dict[str, FACTSDevice] = {}
        self.agc: Optional[AGCInterface] = None
        
        # Reliability & security
        self.contingencies: Dict[str, Contingency] = {}
        self.maintenance: List[MaintenanceSchedule] = []
        
        # Default profiles for load modeling
        self.default_diurnal = [0.6 + 0.4 * math.sin((h-6)/24 * 2 * math.pi) for h in range(24)]
        self.default_weekly = {i: (1.0 if i < 5 else 0.8) for i in range(7)}
        self.default_seasonal = {m: (1.0 + 0.2 * math.cos((m-1)/12 * 2 * math.pi)) for m in range(1, 13)}
        
        logger.info(f"Initializing AdvancedGrid with {len(regions)} regions, {buses_per_region} buses per region")
        
        # Build the grid
        self._build_grid()
        
        logger.info(f"Grid built: {len(self.buses)} buses, {len(self.lines)} lines, {len(self.generators)} generators")

    def _build_grid(self):
        """Build the complete grid infrastructure"""
        self._create_buses()
        self._create_substations()
        self._create_generators()
        self._create_transformers()
        self._create_transmission_lines()
        self._create_interregional_ties()
        self._create_distribution_feeders()
        self._deploy_smart_meters()
        self._deploy_ders()
        self._create_microgrids()
        self._deploy_storage()
        self._deploy_protection()
        self._deploy_pmus()
        self._deploy_facts()
        self._setup_agc()

    def _random_coordinates(self, region: str) -> Tuple[float, float]:
        """Generate random coordinates for a region"""
        # Distribute regions across North America
        region_centers = {
            'A': (49.0, -97.0),  # Manitoba/Saskatchewan
            'B': (53.0, -113.0), # Alberta
            'C': (49.0, -123.0), # British Columbia
            'D': (44.0, -80.0),  # Ontario
            'E': (46.0, -71.0),  # Quebec
            'F': (45.0, -69.0),  # Maritime
            'G': (42.0, -83.0),  # Michigan
            'H': (41.0, -88.0),  # Illinois
            'I': (45.0, -93.0),  # Minnesota
            'J': (47.0, -101.0)  # North Dakota
        }
        
        center_lat, center_lon = region_centers.get(region, (45.0, -95.0))
        
        # Add random offset within ~100km
        lat = center_lat + random.uniform(-0.5, 0.5)
        lon = center_lon + random.uniform(-0.5, 0.5)
        
        return lat, lon

    def _create_buses(self):
        """Create all buses for all regions"""
        logger.info("Creating buses...")
        
        for region in self.regions:
            for i in range(1, self.buses_per_region + 1):
                bus_id = f"{region}{i:03d}"
                lat, lon = self._random_coordinates(region)
                
                # Determine voltage level based on bus number
                if i <= 5:
                    voltage = 765  # EHV
                elif i <= 20:
                    voltage = 345  # HV
                elif i <= 50:
                    voltage = 138  # Sub-transmission
                else:
                    voltage = random.choice([69, 25, 13.8])  # Distribution
                
                bus = Bus(
                    id=bus_id,
                    name=f"Bus {bus_id}",
                    lat=lat,
                    lon=lon,
                    voltage_level=voltage,
                    zone=region
                )
                
                self.buses[bus_id] = bus

    def _create_substations(self):
        """Create substations for each region"""
        from .grid_components import Substation
        
        logger.info("Creating substations...")
        
        for region in self.regions:
            region_buses = [bid for bid in self.buses.keys() if bid.startswith(region)]
            
            # Create 3-5 major substations per region
            num_subs = random.randint(3, 5)
            buses_per_sub = len(region_buses) // num_subs
            
            for i in range(num_subs):
                sub_id = f"SUB_{region}_{i+1:02d}"
                start_idx = i * buses_per_sub
                end_idx = start_idx + buses_per_sub
                sub_buses = region_buses[start_idx:end_idx]
                
                if sub_buses:
                    # Use first bus coordinates as substation location
                    first_bus = self.buses[sub_buses[0]]
                    
                    substation = Substation(
                        id=sub_id,
                        name=f"Substation {region} {i+1}",
                        region=region,
                        lat=first_bus.lat,
                        lon=first_bus.lon,
                        buses=sub_buses,
                        voltage_levels=[765, 345, 138, 69, 25],
                        layout=random.choice(["ring_bus", "breaker_and_a_half", "double_bus"])
                    )
                    
                    self.substations[sub_id] = substation
                    
                    # Update buses with substation reference
                    for bus_id in sub_buses:
                        self.buses[bus_id].substation_id = sub_id

    def _create_generators(self):
        """Create generators with proper sizing to match system load"""
        logger.info("Creating generators with load-balanced sizing...")
        
        # First, estimate total system load
        total_load_mw = 0
        for bus in self.buses.values():
            # Use default load profile to estimate average load per bus
            sample_load = bus.load.total_load(
                datetime.datetime.now(), 
                temp_c=25, 
                econ_idx=1.0, 
                dr_signal=False
            )
            total_load_mw += sample_load
        
        # Target generation = load + losses (5%) + reserve margin (15%)
        target_generation_mw = total_load_mw * 1.20
        logger.info(f"Estimated total load: {total_load_mw:.1f} MW")
        logger.info(f"Target generation capacity: {target_generation_mw:.1f} MW")
        
        fuel_types = ['coal', 'gas', 'nuclear', 'hydro', 'wind', 'solar']
        fuel_probabilities = [0.30, 0.35, 0.15, 0.08, 0.08, 0.04]  # More conventional for stability
        
        # Create profiles for renewables
        wind_profile = [0.3 + 0.2 * math.sin((h + 3) / 24 * 2 * math.pi) for h in range(24)]
        solar_profile = [max(0, math.sin(max(0, h - 6) / 12 * math.pi)) for h in range(24)]
        
        # Select candidate buses for generation (EHV and HV buses)
        candidate_buses = [bus for bus in self.buses.values() 
                          if bus.voltage_level >= 138]
        
        # Calculate required number of generators to meet target
        avg_gen_size = 300  # MW average
        num_generators = max(int(target_generation_mw / avg_gen_size), len(self.regions) * 5)
        
        # Ensure minimum generators per region
        generators_per_region = max(num_generators // len(self.regions), 8)
        logger.info(f"Creating {generators_per_region} generators per region")
        
        current_capacity = 0
        
        for region in self.regions:
            region_buses = [bus for bus in candidate_buses 
                          if bus.id.startswith(region)]
            
            if not region_buses:
                continue
                
            # Create base load generators for this region
            region_generators = 0
            
            while region_generators < generators_per_region and current_capacity < target_generation_mw:
                bus = random.choice(region_buses)
                
                # Skip if bus already has a generator
                if bus.generators:
                    continue
                    
                fuel_type = random.choices(fuel_types, weights=fuel_probabilities)[0]
                
                # Size based on fuel type and system needs
                remaining_need = target_generation_mw - current_capacity
                
                if fuel_type == 'nuclear':
                    capacity = min(random.uniform(800, 1200), remaining_need * 0.15)
                elif fuel_type == 'coal':
                    capacity = min(random.uniform(300, 800), remaining_need * 0.12)
                elif fuel_type == 'gas':
                    capacity = min(random.uniform(100, 500), remaining_need * 0.10)
                elif fuel_type == 'hydro':
                    capacity = min(random.uniform(50, 300), remaining_need * 0.08)
                elif fuel_type in ['wind', 'solar']:
                    capacity = min(random.uniform(50, 200), remaining_need * 0.06)
                else:
                    capacity = min(random.uniform(100, 400), remaining_need * 0.10)
                
                gen_id = f"GEN_{bus.id}"
                
                generator = Generator(
                    id=gen_id,
                    name=f"Generator {bus.id}",
                    bus_id=bus.id,
                    type=fuel_type,
                    capacity_mw=capacity,
                    min_up_time_hr=random.uniform(1, 8),
                    min_down_time_hr=random.uniform(1, 4),
                    ramp_rate_mw_per_min=capacity * random.uniform(0.01, 0.05),
                    startup_cost=random.uniform(5000, 50000),
                    heat_rate=random.uniform(8000, 12000),
                    fuel_cost=random.uniform(2, 8),
                    variable_om=random.uniform(2, 8),
                    cf_profile=wind_profile if fuel_type == 'wind' else solar_profile if fuel_type == 'solar' else None,
                    reservoir_level=random.uniform(60, 95) if fuel_type == 'hydro' else None,
                    is_baseload=fuel_type in ['nuclear', 'coal'],
                    is_peaking=fuel_type == 'gas'
                )
                
                self.generators[gen_id] = generator
                bus.generators.append(gen_id)
                current_capacity += capacity
                region_generators += 1
        
        actual_capacity = sum(gen.capacity_mw for gen in self.generators.values())
        logger.info(f"Total generation capacity created: {actual_capacity:.1f} MW")
        logger.info(f"Reserve margin: {((actual_capacity - total_load_mw) / total_load_mw * 100):.1f}%")

    def _create_transformers(self):
        """Create transformers for voltage level changes"""
        logger.info("Creating transformers...")
        
        voltage_pairs = [
            (765, 345), (345, 138), (138, 69), 
            (69, 25), (25, 13.8), (138, 25)
        ]
        
        for sub_id, substation in self.substations.items():
            # Create transformers for each voltage level pair in substation
            available_levels = substation.voltage_levels
            
            for from_v, to_v in voltage_pairs:
                if from_v in available_levels and to_v in available_levels:
                    # Find buses at these voltage levels
                    from_buses = [bid for bid in substation.buses 
                                 if self.buses[bid].voltage_level == from_v]
                    to_buses = [bid for bid in substation.buses 
                               if self.buses[bid].voltage_level == to_v]
                    
                    if from_buses and to_buses:
                        xfmr_id = f"XFMR_{sub_id}_{from_v}_{to_v}"
                        
                        transformer = Transformer(
                            id=xfmr_id,
                            name=f"Transformer {sub_id} {from_v}-{to_v}kV",
                            from_bus=from_buses[0],
                            to_bus=to_buses[0],
                            from_level=from_v,
                            to_level=to_v,
                            capacity_mva=random.uniform(100, 1000),
                            impedance_pu=random.uniform(0.08, 0.15)
                        )
                        
                        self.transformers[xfmr_id] = transformer
                        substation.transformers.append(xfmr_id)

    def _create_transmission_lines(self):
        """Create simplified but robust transmission network topology"""
        logger.info("Creating simplified transmission network...")
        
        for region in self.regions:
            region_buses = [self.buses[bid] for bid in self.buses.keys() 
                           if bid.startswith(region) and self.buses[bid].voltage_level >= 138]
            
            if len(region_buses) < 2:
                continue
                
            # Create ring topology with cross-connections (N-1 secure)
            for i in range(len(region_buses)):
                # Ring connection to next bus
                next_bus = region_buses[(i + 1) % len(region_buses)]
                self._create_line(region_buses[i], next_bus)
                
                # Add selective cross-connections for redundancy (every 3rd bus)
                if i % 3 == 0 and len(region_buses) > 6:
                    cross_idx = (i + len(region_buses) // 2) % len(region_buses)
                    if cross_idx != i and cross_idx != (i + 1) % len(region_buses):
                        self._create_line(region_buses[i], region_buses[cross_idx])

    def _create_interregional_ties(self):
        """Create high-capacity tie lines between regions"""
        logger.info("Creating interregional ties...")
        
        for i in range(len(self.regions)):
            for j in range(i + 1, len(self.regions)):
                region_a = self.regions[i]
                region_b = self.regions[j]
                
                # Find EHV buses in each region
                buses_a = [self.buses[bid] for bid in self.buses.keys() 
                          if bid.startswith(region_a) and self.buses[bid].voltage_level >= 345]
                buses_b = [self.buses[bid] for bid in self.buses.keys() 
                          if bid.startswith(region_b) and self.buses[bid].voltage_level >= 345]
                
                if buses_a and buses_b:
                    # Create 1-2 tie lines between regions
                    num_ties = random.randint(1, 2)
                    for _ in range(num_ties):
                        bus_a = random.choice(buses_a)
                        bus_b = random.choice(buses_b)
                        self._create_line(bus_a, bus_b, is_tie=True)

    def _create_line(self, bus_a: Bus, bus_b: Bus, is_tie: bool = False):
        """Create a transmission line between two buses"""
        voltage_kv = min(bus_a.voltage_level, bus_b.voltage_level)
        
        # Capacity based on voltage level
        capacity_map = {
            765: random.uniform(2000, 4000),
            500: random.uniform(1500, 3000),
            345: random.uniform(1000, 2000),
            230: random.uniform(500, 1500),
            138: random.uniform(200, 800),
            115: random.uniform(150, 600),
            69: random.uniform(50, 200)
        }
        
        capacity = capacity_map.get(voltage_kv, 200)
        if is_tie:
            capacity *= 1.5  # Tie lines typically have higher capacity
        
        line = EHVLine.create_line(
            bus_a, bus_b, voltage_kv, capacity,
            terrain_factor=random.uniform(1.0, 2.0),
            row_restriction=random.choice([None, "environmental", "urban"]),
            transfer_limit_mw=capacity * 0.9,
            is_tie_line=is_tie
        )
        
        self.lines[line.id] = line

    # Stub methods for remaining components
    def _create_distribution_feeders(self):
        logger.info("Creating distribution feeders...")
        # Implementation would be similar to above

    def _deploy_smart_meters(self):
        logger.info("Deploying smart meters...")
        # Implementation would create smart meters for all buses

    def _deploy_ders(self):
        logger.info("Deploying DERs...")
        # Implementation would create distributed energy resources

    def _create_microgrids(self):
        logger.info("Creating microgrids...")
        # Implementation would group DERs into microgrids

    def _deploy_storage(self):
        logger.info("Deploying energy storage...")
        # Implementation would create storage systems

    def _deploy_protection(self):
        logger.info("Deploying protection systems...")
        # Implementation would create protection relays

    def _deploy_pmus(self):
        logger.info("Deploying PMUs...")
        # Implementation would create PMUs at strategic locations

    def _deploy_facts(self):
        logger.info("Deploying FACTS devices...")
        # Implementation would create FACTS devices

    def _setup_agc(self):
        logger.info("Setting up AGC...")
        # Implementation would setup AGC interface

    def to_json(self) -> str:
        """Export grid to JSON"""
        logger.info("Exporting grid to JSON...")
        
        grid_data = {
            "metadata": {
                "regions": self.regions,
                "buses_per_region": self.buses_per_region,
                "total_buses": len(self.buses),
                "total_lines": len(self.lines),
                "total_generators": len(self.generators),
                "created_at": datetime.datetime.now().isoformat()
            },
            "buses": {bid: bus.to_dict() for bid, bus in self.buses.items()},
            "generators": {gid: gen.to_dict() for gid, gen in self.generators.items()},
            "lines": {lid: line.to_dict() for lid, line in self.lines.items()},
            "transformers": {tid: xfmr.to_dict() for tid, xfmr in self.transformers.items()},
            "substations": {sid: sub.to_dict() for sid, sub in self.substations.items()}
        }
        
        return json.dumps(grid_data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'AdvancedGrid':
        """Load grid from JSON"""
        logger.info("Loading grid from JSON...")
        
        data = json.loads(json_str)
        metadata = data["metadata"]
        
        # Create empty grid
        grid = cls.__new__(cls)
        grid.regions = metadata["regions"]
        grid.buses_per_region = metadata["buses_per_region"]
        
        # Initialize containers
        grid.buses = {}
        grid.generators = {}
        grid.lines = {}
        grid.transformers = {}
        grid.substations = {}
        
        # Load basic data (simplified)
        for bid, bus_data in data.get("buses", {}).items():
            grid.buses[bid] = Bus(**bus_data)
        
        for gid, gen_data in data.get("generators", {}).items():
            grid.generators[gid] = Generator(**gen_data)
            
        for lid, line_data in data.get("lines", {}).items():
            grid.lines[lid] = EHVLine(**line_data)
        
        logger.info(f"Loaded grid: {len(grid.buses)} buses, {len(grid.lines)} lines")
        
        return grid

    def get_summary(self) -> Dict[str, Any]:
        """Get grid summary statistics"""
        return {
            "total_buses": len(self.buses),
            "total_generators": len(self.generators),
            "total_capacity_mw": sum(gen.capacity_mw for gen in self.generators.values()),
            "total_lines": len(self.lines),
            "total_substations": len(self.substations),
            "regions": self.regions,
            "voltage_levels": list(set(bus.voltage_level for bus in self.buses.values()))
        } 