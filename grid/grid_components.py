"""
Additional Grid Components - Part 2 of Advanced Grid Implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from dataclasses_json import dataclass_json
import datetime

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