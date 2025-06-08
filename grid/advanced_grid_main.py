"""
Main AdvancedGrid Class - Comprehensive Grid Container
Implements the full 2000+ bus system with all modeling capabilities
"""

import json
import random
import string
import math
import datetime
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger

from .advanced_grid import *
from .grid_components import *

class AdvancedGrid:
    """
    Comprehensive grid container with 2000+ buses
    Implements the full blended topology specification
    """
    
    def __init__(self, regions: List[str], buses_per_region: int = 200):
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
        """Create generators across the grid"""
        logger.info("Creating generators...")
        
        fuel_types = ['coal', 'gas', 'nuclear', 'hydro', 'wind', 'solar']
        fuel_probabilities = [0.25, 0.30, 0.10, 0.10, 0.15, 0.10]
        
        # Create profiles for renewables
        wind_profile = [0.3 + 0.2 * math.sin((h + 3) / 24 * 2 * math.pi) for h in range(24)]
        solar_profile = [max(0, math.sin(max(0, h - 6) / 12 * math.pi)) for h in range(24)]
        
        for bus_id, bus in self.buses.items():
            # Only add generators to certain buses (transmission level mainly)
            if bus.voltage_level >= 138 and random.random() < 0.15:
                fuel_type = random.choices(fuel_types, weights=fuel_probabilities)[0]
                
                # Size based on fuel type and voltage level
                if fuel_type == 'nuclear':
                    capacity = random.uniform(800, 1200)
                elif fuel_type == 'coal':
                    capacity = random.uniform(300, 800)
                elif fuel_type == 'gas':
                    capacity = random.uniform(100, 500)
                elif fuel_type == 'hydro':
                    capacity = random.uniform(50, 300)
                elif fuel_type in ['wind', 'solar']:
                    capacity = random.uniform(50, 200)
                else:
                    capacity = random.uniform(100, 400)
                
                gen_id = f"GEN_{bus_id}"
                
                generator = Generator(
                    id=gen_id,
                    name=f"Generator {bus_id}",
                    bus_id=bus_id,
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
        """Create transmission lines within and between regions"""
        logger.info("Creating transmission lines...")
        
        for region in self.regions:
            region_buses = [self.buses[bid] for bid in self.buses.keys() 
                           if bid.startswith(region) and self.buses[bid].voltage_level >= 138]
            
            # Create intra-region mesh
            for i in range(len(region_buses)):
                for j in range(i + 1, min(i + 4, len(region_buses))):  # Connect to nearby buses
                    if random.random() < 0.3:  # Not all possible connections
                        self._create_line(region_buses[i], region_buses[j])
            
            # Create radial connections for some buses
            for i in range(5, len(region_buses)):
                if random.random() < 0.4:
                    target_idx = random.randint(0, min(i-1, 10))
                    self._create_line(region_buses[i], region_buses[target_idx])

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

    def _create_distribution_feeders(self):
        """Create distribution feeders from substations"""
        logger.info("Creating distribution feeders...")
        
        for sub_id, substation in self.substations.items():
            # Create 5-10 feeders per substation
            num_feeders = random.randint(5, 10)
            
            for i in range(num_feeders):
                feeder_id = f"FDR_{sub_id}_{i+1:02d}"
                
                # Find distribution buses in this substation
                dist_buses = [bid for bid in substation.buses 
                             if self.buses[bid].voltage_level <= 69]
                
                if dist_buses:
                    feeder_buses = random.sample(dist_buses, min(len(dist_buses), 20))
                    
                    feeder = Feeder(
                        id=feeder_id,
                        name=f"Feeder {sub_id} {i+1}",
                        region=substation.region,
                        substation_id=sub_id,
                        voltage_kv=random.choice([4, 12, 25, 35]),
                        buses=feeder_buses,
                        peak_load_mw=sum(self.buses[bid].load.total_load(
                            datetime.datetime.now(), 25, 1.0, False) for bid in feeder_buses)
                    )
                    
                    self.feeders[feeder_id] = feeder

    def _deploy_smart_meters(self):
        """Deploy smart meters at all buses"""
        logger.info("Deploying smart meters...")
        
        for bus_id in self.buses.keys():
            meter_id = f"SM_{bus_id}"
            meter = SmartMeter(
                id=meter_id,
                name=f"Smart Meter {bus_id}",
                bus_id=bus_id,
                communication=random.choice(["RF_mesh", "PLC", "cellular"])
            )
            self.smart_meters[meter_id] = meter

    def _deploy_ders(self):
        """Deploy distributed energy resources"""
        logger.info("Deploying DERs...")
        
        der_types = ['solar', 'wind', 'BESS', 'fuel_cell']
        
        for region in self.regions:
            region_buses = [bid for bid in self.buses.keys() 
                           if bid.startswith(region) and self.buses[bid].voltage_level <= 69]
            
            # Deploy 20-50 DERs per region
            num_ders = random.randint(20, 50)
            
            for i in range(num_ders):
                der_id = f"DER_{region}_{i+1:03d}"
                bus_id = random.choice(region_buses)
                der_type = random.choice(der_types)
                
                der = DER(
                    id=der_id,
                    name=f"DER {region} {i+1}",
                    region=region,
                    bus_id=bus_id,
                    type=der_type,
                    capacity_kw=random.uniform(10, 500),
                    can_island=(der_type == 'BESS')
                )
                
                self.ders[der_id] = der

    def _create_microgrids(self):
        """Create microgrid systems"""
        logger.info("Creating microgrids...")
        
        for region in self.regions:
            # Create 2-5 microgrids per region
            num_microgrids = random.randint(2, 5)
            
            region_ders = [did for did in self.ders.keys() if did.startswith(f"DER_{region}")]
            region_buses = list(set(self.ders[did].bus_id for did in region_ders))
            
            for i in range(num_microgrids):
                mg_id = f"MG_{region}_{i+1:02d}"
                
                # Select DERs for this microgrid
                mg_ders = random.sample(region_ders, min(len(region_ders), 8))
                mg_buses = list(set(self.ders[did].bus_id for did in mg_ders))
                
                microgrid = Microgrid(
                    id=mg_id,
                    name=f"Microgrid {region} {i+1}",
                    region=region,
                    bus_ids=mg_buses,
                    der_ids=mg_ders,
                    backup_generation_mw=sum(self.ders[did].capacity_kw/1000 
                                           for did in mg_ders if self.ders[did].type in ['BESS', 'fuel_cell']),
                    critical_loads_mw=random.uniform(5, 20)
                )
                
                self.microgrids[mg_id] = microgrid

    def _deploy_storage(self):
        """Deploy energy storage systems"""
        logger.info("Deploying energy storage...")
        
        storage_types = ['lithium_ion', 'pumped_hydro', 'compressed_air']
        
        for region in self.regions:
            # Deploy 3-8 storage systems per region
            num_storage = random.randint(3, 8)
            region_buses = [bid for bid in self.buses.keys() if bid.startswith(region)]
            
            for i in range(num_storage):
                ess_id = f"ESS_{region}_{i+1:02d}"
                bus_id = random.choice(region_buses)
                technology = random.choice(storage_types)
                
                # Size based on technology
                if technology == 'pumped_hydro':
                    capacity = random.uniform(500, 2000)
                    power = capacity * random.uniform(0.2, 0.4)
                elif technology == 'compressed_air':
                    capacity = random.uniform(100, 500)
                    power = capacity * random.uniform(0.3, 0.5)
                else:  # lithium_ion
                    capacity = random.uniform(10, 200)
                    power = capacity * random.uniform(0.5, 1.0)
                
                storage = EnergyStorage(
                    id=ess_id,
                    name=f"Storage {region} {i+1}",
                    bus_id=bus_id,
                    technology=technology,
                    capacity_mwh=capacity,
                    power_rating_mw=power,
                    efficiency=0.85 if technology == 'lithium_ion' else 0.75
                )
                
                self.storage[ess_id] = storage

    def _deploy_protection(self):
        """Deploy protection relays"""
        logger.info("Deploying protection systems...")
        
        # Deploy relays at substations and major buses
        for sub_id, substation in self.substations.items():
            for i, bus_id in enumerate(substation.buses[:5]):  # Protect major buses
                relay_id = f"PR_{sub_id}_{i+1:02d}"
                
                relay = ProtectionRelay(
                    id=relay_id,
                    name=f"Relay {sub_id} {i+1}",
                    region=substation.region,
                    location_id=bus_id,
                    settings={
                        'overcurrent_pickup': random.uniform(1.2, 1.5),
                        'undervoltage_pickup': random.uniform(0.85, 0.95),
                        'time_delay': random.uniform(0.1, 0.5)
                    }
                )
                
                self.relays[relay_id] = relay

    def _deploy_pmus(self):
        """Deploy PMUs for wide area monitoring"""
        logger.info("Deploying PMUs...")
        
        # Deploy PMUs at strategic locations
        for region in self.regions:
            region_buses = [bid for bid in self.buses.keys() 
                           if bid.startswith(region) and self.buses[bid].voltage_level >= 138]
            
            # Deploy 3-6 PMUs per region
            num_pmus = min(random.randint(3, 6), len(region_buses))
            selected_buses = random.sample(region_buses, num_pmus)
            
            for i, bus_id in enumerate(selected_buses):
                pmu_id = f"PMU_{region}_{i+1:02d}"
                
                pmu = PMU(
                    id=pmu_id,
                    name=f"PMU {region} {i+1}",
                    region=region,
                    location_id=bus_id,
                    sample_rate_hz=random.choice([30, 60, 120])
                )
                
                self.pmus[pmu_id] = pmu

    def _deploy_facts(self):
        """Deploy FACTS devices"""
        logger.info("Deploying FACTS devices...")
        
        facts_types = ['SVC', 'STATCOM', 'TCSC', 'UPFC']
        
        # Deploy FACTS at strategic locations
        major_buses = [bid for bid in self.buses.keys() 
                      if self.buses[bid].voltage_level >= 345]
        
        num_facts = min(len(major_buses) // 4, 20)  # Don't overdo it
        selected_buses = random.sample(major_buses, num_facts)
        
        for i, bus_id in enumerate(selected_buses):
            facts_id = f"FACTS_{i+1:03d}"
            device_type = random.choice(facts_types)
            
            facts = FACTSDevice(
                id=facts_id,
                name=f"FACTS {device_type} {i+1}",
                type=device_type,
                location_id=bus_id,
                rating_mvar=random.uniform(100, 500),
                settings={
                    'voltage_setpoint': random.uniform(1.0, 1.05),
                    'reactive_limit': random.uniform(200, 400)
                }
            )
            
            self.facts_devices[facts_id] = facts

    def _setup_agc(self):
        """Setup AGC interface"""
        logger.info("Setting up AGC...")
        
        # Find all generators suitable for AGC
        agc_plants = [gid for gid, gen in self.generators.items() 
                     if gen.type in ['gas', 'hydro'] and gen.capacity_mw >= 50]
        
        total_capacity = sum(self.generators[gid].capacity_mw for gid in agc_plants)
        
        self.agc = AGCInterface(
            id="AGC_MAIN",
            name="Main AGC System",
            balancing_area="MAIN_BA",
            participating_plants=agc_plants,
            regulation_capacity_mw=total_capacity * 0.1  # 10% for regulation
        )

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
            "substations": {sid: sub.to_dict() for sid, sub in self.substations.items()},
            "feeders": {fid: feeder.to_dict() for fid, feeder in self.feeders.items()},
            "ders": {did: der.to_dict() for did, der in self.ders.items()},
            "microgrids": {mid: mg.to_dict() for mid, mg in self.microgrids.items()},
            "storage": {eid: ess.to_dict() for eid, ess in self.storage.items()},
            "contingencies": {cid: cont.to_dict() for cid, cont in self.contingencies.items()},
            "pmus": {pid: pmu.to_dict() for pid, pmu in self.pmus.items()},
            "facts": {fid: facts.to_dict() for fid, facts in self.facts_devices.items()}
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
        
        # Initialize all containers
        grid.buses = {}
        grid.generators = {}
        grid.lines = {}
        grid.transformers = {}
        grid.substations = {}
        grid.feeders = {}
        grid.ders = {}
        grid.microgrids = {}
        grid.storage = {}
        grid.contingencies = {}
        grid.pmus = {}
        grid.facts_devices = {}
        grid.smart_meters = {}
        grid.relays = {}
        grid.maintenance = []
        grid.agc = None
        
        # Load data from JSON (simplified - full implementation would use from_dict methods)
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
            "total_feeders": len(self.feeders),
            "total_ders": len(self.ders),
            "total_microgrids": len(self.microgrids),
            "total_storage": len(self.storage),
            "total_storage_capacity_mwh": sum(ess.capacity_mwh for ess in self.storage.values()),
            "regions": self.regions,
            "voltage_levels": list(set(bus.voltage_level for bus in self.buses.values()))
        } 