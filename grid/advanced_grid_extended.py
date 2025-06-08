# Continuation of advanced_grid.py - Additional Components

@dataclass_json
@dataclass
class Substation:
    """Comprehensive substation model"""
    id: str
    name: str
    region: str
    lat: float
    lon: float
    buses: List[str] = field(default_factory=list)
    voltage_levels: List[int] = field(default_factory=list)
    layout: str = "ring_bus"  # ring_bus, breaker_and_a_half, double_bus
    transformers: List[str] = field(default_factory=list)
    tie_lines: List[str] = field(default_factory=list)
    protection_zones: List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class Feeder:
    """Distribution feeder model"""
    id: str
    name: str
    region: str
    substation_id: str
    voltage_kv: int  # 4-35 kV
    buses: List[str] = field(default_factory=list)
    automated_reclosers: bool = True
    sectionalizers: bool = True
    peak_load_mw: float = 0.0
    length_km: float = 0.0

@dataclass_json
@dataclass
class DER:
    """Distributed Energy Resource"""
    id: str
    name: str
    region: str
    bus_id: str
    type: str  # solar, wind, BESS, fuel_cell
    capacity_kw: float
    interconnection_level: str = "distribution"  # distribution, transmission
    inverter_type: str = "grid_following"
    can_island: bool = False

@dataclass_json
@dataclass
class Microgrid:
    """Microgrid system"""
    id: str
    name: str
    region: str
    bus_ids: List[str] = field(default_factory=list)
    der_ids: List[str] = field(default_factory=list)
    can_island: bool = True
    backup_generation_mw: float = 0.0
    critical_loads_mw: float = 0.0

@dataclass_json
@dataclass
class ProtectionRelay:
    """Protection relay system"""
    id: str
    name: str
    region: str
    location_id: str  # bus or line ID
    type: str = "digital"  # digital, electromechanical
    settings: Dict[str, float] = field(default_factory=dict)
    communication: bool = True

@dataclass_json
@dataclass
class PMU:
    """Phasor Measurement Unit"""
    id: str
    name: str
    region: str
    location_id: str
    sample_rate_hz: float = 60.0
    gps_sync: bool = True
    data_concentrator: Optional[str] = None

@dataclass_json
@dataclass
class AGCInterface:
    """Automatic Generation Control Interface"""
    id: str
    name: str
    balancing_area: str
    participating_plants: List[str] = field(default_factory=list)
    regulation_capacity_mw: float = 0.0
    response_time_sec: float = 60.0

@dataclass_json
@dataclass
class SmartMeter:
    """Smart meter for load monitoring"""
    id: str
    name: str
    bus_id: str
    communication: str = "RF_mesh"  # RF_mesh, PLC, cellular
    data_interval_min: int = 15

# ========================= Market & Economic =========================

@dataclass_json
@dataclass
class CostCurve:
    """Generator cost curve"""
    generator_id: str
    segments: List[Tuple[float, float]]  # (MW, $/MWh) points
    startup_cost: float = 0.0
    no_load_cost: float = 0.0

@dataclass_json
@dataclass
class LMP:
    """Locational Marginal Price"""
    bus_id: str
    timestamp: datetime.datetime
    energy_price: float  # $/MWh
    congestion_price: float = 0.0
    loss_price: float = 0.0

# ========================= Advanced Controls =========================

@dataclass_json
@dataclass
class FACTSDevice:
    """Flexible AC Transmission System device"""
    id: str
    name: str
    type: str  # SVC, STATCOM, TCSC, UPFC
    location_id: str  # bus or line ID
    rating_mvar: float = 0.0
    settings: Dict[str, float] = field(default_factory=dict)
    automatic_control: bool = True

@dataclass_json
@dataclass
class EnergyStorage:
    """Energy storage system"""
    id: str
    name: str
    bus_id: str
    technology: str  # lithium_ion, pumped_hydro, compressed_air
    capacity_mwh: float
    power_rating_mw: float
    efficiency: float = 0.85
    state_of_charge: float = 0.5  # 0-1
    min_soc: float = 0.1
    max_soc: float = 0.9

# ========================= Contingency & Reliability =========================

@dataclass_json
@dataclass
class Contingency:
    """Contingency scenario definition"""
    id: str
    name: str
    type: str  # N-1, N-2, N-k
    elements: List[str]  # list of element IDs to remove
    probability: float = 0.0
    severity_dist: Dict[str, Any] = field(default_factory=dict)
    duration_min: float = 60.0  # expected outage duration

@dataclass_json
@dataclass
class MaintenanceSchedule:
    """Planned maintenance schedule"""
    element_id: str
    element_type: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    description: str = ""
    impact_assessment: Dict[str, Any] = field(default_factory=dict)

# ========================= Cyber-Physical Security =========================

@dataclass_json
@dataclass
class CyberZone:
    """Cybersecurity zone"""
    id: str
    name: str
    security_level: str  # high, medium, low
    assets: List[str] = field(default_factory=list)
    firewalls: List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class SecureCommLink:
    """Secure communication link"""
    id: str
    name: str
    type: str  # fiber, microwave, satellite
    endpoints: Tuple[str, str]
    encryption: bool = True
    redundancy: bool = False

# ========================= Environmental & Regulatory =========================

@dataclass_json
@dataclass
class EmissionsData:
    """Emissions data for generators"""
    generator_id: str
    co2_rate: float  # tons/MWh
    nox_rate: float  # tons/MWh
    so2_rate: float  # tons/MWh
    emission_limits: Dict[str, float] = field(default_factory=dict)

# ========================= Main Grid Container =========================

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
        
        # Market & economics
        self.cost_curves: Dict[str, CostCurve] = {}
        self.lmp_data: Dict[str, List[LMP]] = {}
        
        # Reliability & security
        self.contingencies: Dict[str, Contingency] = {}
        self.maintenance: List[MaintenanceSchedule] = []
        self.cyber_zones: Dict[str, CyberZone] = {}
        self.comm_links: Dict[str, SecureCommLink] = {}
        
        # Environmental
        self.emissions: Dict[str, EmissionsData] = {}
        
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
        self._initialize_market_data()
        self._setup_cyber_security()
    
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
            "contingencies": {cid: cont.to_dict() for cid, cont in self.contingencies.items()}
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
        grid.feeders = {}
        grid.ders = {}
        grid.microgrids = {}
        grid.storage = {}
        grid.contingencies = {}
        
        # Load data
        grid.buses = {bid: Bus.from_dict(bus_data) for bid, bus_data in data["buses"].items()}
        grid.generators = {gid: Generator.from_dict(gen_data) for gid, gen_data in data["generators"].items()}
        grid.lines = {lid: EHVLine.from_dict(line_data) for lid, line_data in data["lines"].items()}
        
        logger.info(f"Loaded grid: {len(grid.buses)} buses, {len(grid.lines)} lines")
        
        return grid 