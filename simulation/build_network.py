"""
Network Builder - Convert AdvancedGrid to Pandapower Network

This module builds pandapower networks from our AdvancedGrid topology
for power flow analysis and contingency studies.
"""

import pandapower as pp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
import datetime

from grid import AdvancedGrid, Bus, Generator, EHVLine, Transformer

class NetworkBuilder:
    """
    Converts AdvancedGrid topology to pandapower network
    """
    
    def __init__(self):
        self.bus_mapping: Dict[str, int] = {}  # AdvancedGrid bus ID -> pandapower bus index
        self.gen_mapping: Dict[str, int] = {}  # AdvancedGrid gen ID -> pandapower gen index
        self.line_mapping: Dict[str, int] = {}  # AdvancedGrid line ID -> pandapower line index
        self.trafo_mapping: Dict[str, int] = {}  # AdvancedGrid trafo ID -> pandapower trafo index
        
    def build_network(self, grid: AdvancedGrid, 
                     timestamp: Optional[datetime.datetime] = None,
                     temperature_c: float = 25.0,
                     economic_index: float = 1.0,
                     dr_signal: bool = False) -> pp.pandapowerNet:
        """
        Build pandapower network from AdvancedGrid
        
        Args:
            grid: AdvancedGrid topology
            timestamp: Time for load/generation calculations
            temperature_c: Temperature for load adjustment
            economic_index: Economic index for industrial load
            dr_signal: Demand response signal
            
        Returns:
            pandapower network ready for power flow
        """
        logger.info("Building pandapower network from AdvancedGrid...")
        
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Create empty network
        net = pp.create_empty_network(f_hz=60.0, sn_mva=100.0)
        
        # Build network components
        self._add_buses(net, grid)
        self._add_loads(net, grid, timestamp, temperature_c, economic_index, dr_signal)
        self._add_generators(net, grid, timestamp)
        self._add_lines(net, grid)
        self._add_transformers(net, grid)
        self._add_external_grid(net, grid)
        
        logger.info(f"Network built: {len(net.bus)} buses, {len(net.line)} lines, "
                   f"{len(net.gen)} generators, {len(net.load)} loads")
        
        return net
    
    def _add_buses(self, net: pp.pandapowerNet, grid: AdvancedGrid):
        """Add all buses to the network"""
        logger.info("Adding buses...")
        
        for bus_id, bus in grid.buses.items():
            # Convert to appropriate base voltage
            vn_kv = self._get_base_voltage(bus.voltage_level)
            
            pp_bus_idx = pp.create_bus(
                net,
                vn_kv=vn_kv,
                name=bus.name,
                index=None,
                geodata=(bus.lat, bus.lon),
                type="b",  # busbar
                zone=bus.zone
            )
            
            self.bus_mapping[bus_id] = pp_bus_idx
    
    def _get_base_voltage(self, voltage_kv: float) -> float:
        """Convert voltage level to standard base voltage"""
        # Map to standard voltage levels
        voltage_map = {
            765: 765.0,
            500: 500.0,
            345: 345.0,
            230: 230.0,
            138: 138.0,
            115: 115.0,
            69: 69.0,
            25: 25.0,
            13.8: 13.8,
            4.16: 4.16
        }
        
        # Find closest standard voltage
        closest = min(voltage_map.keys(), key=lambda x: abs(x - voltage_kv))
        return voltage_map[closest]
    
    def _add_loads(self, net: pp.pandapowerNet, grid: AdvancedGrid,
                   timestamp: datetime.datetime, temperature_c: float,
                   economic_index: float, dr_signal: bool):
        """Add loads to all buses"""
        logger.info("Adding loads...")
        
        for bus_id, bus in grid.buses.items():
            if bus.load is not None:
                # Calculate total load for this timestamp
                p_mw = bus.load.total_load(timestamp, temperature_c, economic_index, dr_signal)
                
                # Estimate reactive power (typical power factor 0.95)
                q_mvar = p_mw * np.tan(np.arccos(0.95))
                
                pp_bus_idx = self.bus_mapping[bus_id]
                
                pp.create_load(
                    net,
                    bus=pp_bus_idx,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    name=f"Load_{bus.name}",
                    scaling=1.0,
                    const_z_percent=0.0,
                    const_i_percent=0.0,
                    controllable=dr_signal  # Controllable if DR is active
                )
    
    def _add_generators(self, net: pp.pandapowerNet, grid: AdvancedGrid,
                       timestamp: datetime.datetime):
        """Add generators to the network"""
        logger.info("Adding generators...")
        
        for gen_id, gen in grid.generators.items():
            pp_bus_idx = self.bus_mapping[gen.bus_id]
            
            # Get available capacity for this timestamp
            max_p_mw = gen.available_capacity(timestamp)
            
            # Initial dispatch - simplified economic dispatch
            if gen.is_baseload:
                p_mw = max_p_mw * 0.9  # Run baseload at 90%
            elif gen.is_peaking:
                p_mw = max_p_mw * 0.3  # Run peaking at 30%
            else:
                p_mw = max_p_mw * 0.6  # Intermediate units at 60%
            
            # Generator costs
            cost_per_mw = gen.marginal_cost()
            
            # Check if this bus already has a voltage controlling element
            existing_gens = net.gen[net.gen['bus'] == pp_bus_idx]
            existing_ext_grids = net.ext_grid[net.ext_grid['bus'] == pp_bus_idx]
            
            # Set voltage setpoint based on existing elements
            if len(existing_gens) > 0 or len(existing_ext_grids) > 0:
                # If there's already a voltage controlling element, make this one PQ
                vm_pu = np.nan  # PQ mode
            else:
                # First generator on this bus can control voltage
                vm_pu = 1.01
            
            pp_gen_idx = pp.create_gen(
                net,
                bus=pp_bus_idx,
                p_mw=p_mw,
                vm_pu=vm_pu,
                name=gen.name,
                max_p_mw=max_p_mw,
                min_p_mw=0.0,
                max_q_mvar=max_p_mw * 0.5,  # Typical reactive capability
                min_q_mvar=-max_p_mw * 0.3,
                controllable=True,
                type=gen.type
            )
            
            self.gen_mapping[gen_id] = pp_gen_idx
            
            # Add cost curve
            pp.create_poly_cost(
                net,
                element=pp_gen_idx,
                et="gen",
                cp1_eur_per_mw=cost_per_mw,
                cp0_eur=gen.startup_cost if gen.online else 0
            )
    
    def _add_lines(self, net: pp.pandapowerNet, grid: AdvancedGrid):
        """Add transmission lines"""
        logger.info("Adding transmission lines...")
        
        for line_id, line in grid.lines.items():
            from_bus = self.bus_mapping[line.from_bus]
            to_bus = self.bus_mapping[line.to_bus]
            
            # Calculate line parameters
            r_ohm_per_km = line.r_ohm_per_km * line.terrain_factor
            x_ohm_per_km = line.x_ohm_per_km * line.terrain_factor
            c_nf_per_km = line.c_nf_per_km
            
            # Maximum current based on thermal rating
            # I = P / (sqrt(3) * V * cos(phi))
            cos_phi = 0.95
            max_i_ka = line.rating_normal / (np.sqrt(3) * line.voltage_kv * cos_phi) * 1000
            
            pp_line_idx = pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=line.length_km,
                name=line.name,
                r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=x_ohm_per_km,
                c_nf_per_km=c_nf_per_km,
                max_i_ka=max_i_ka,
                type="ol",  # overhead line
                in_service=True
            )
            
            self.line_mapping[line_id] = pp_line_idx
    
    def _add_transformers(self, net: pp.pandapowerNet, grid: AdvancedGrid):
        """Add transformers"""
        logger.info("Adding transformers...")
        
        for trafo_id, trafo in grid.transformers.items():
            hv_bus = self.bus_mapping[trafo.from_bus]
            lv_bus = self.bus_mapping[trafo.to_bus]
            
            # Get voltage levels
            hv_kv = self._get_base_voltage(trafo.from_level)
            lv_kv = self._get_base_voltage(trafo.to_level)
            
            # Calculate transformer parameters
            vk_percent = trafo.impedance_pu * 100  # Convert to percentage
            vkr_percent = vk_percent * 0.1  # Assume R is 10% of total impedance
            
            # Maximum loading based on thermal rating
            max_loading_percent = 100.0
            
            pp_trafo_idx = pp.create_transformer_from_parameters(
                net,
                hv_bus=hv_bus,
                lv_bus=lv_bus,
                name=trafo.name,
                sn_mva=trafo.capacity_mva,
                vn_hv_kv=hv_kv,
                vn_lv_kv=lv_kv,
                vk_percent=vk_percent,
                vkr_percent=vkr_percent,
                pfe_kw=trafo.capacity_mva * 1.0,  # 1% iron losses
                i0_percent=0.5,  # 0.5% no-load current
                tap_side="hv",
                tap_neutral=0,
                tap_min=-10,
                tap_max=10,
                tap_step_percent=1.25,
                tap_pos=0,
                max_loading_percent=max_loading_percent
            )
            
            self.trafo_mapping[trafo_id] = pp_trafo_idx
    
    def _add_external_grid(self, net: pp.pandapowerNet, grid: AdvancedGrid):
        """Add external grid connections (slack buses)"""
        logger.info("Adding external grid connections...")
        
        # Find the highest voltage buses in each region for slack buses
        slack_buses = []
        for region in grid.regions:
            region_buses = [(bid, bus) for bid, bus in grid.buses.items() 
                           if bid.startswith(region)]
            
            if region_buses:
                # Find the highest voltage bus in this region
                highest_v_bus = max(region_buses, key=lambda x: x[1].voltage_level)
                slack_buses.append(highest_v_bus)
        
        # Create external grid at the first slack bus (main slack)
        if slack_buses:
            main_slack_bus_id, main_slack_bus = slack_buses[0]
            pp_bus_idx = self.bus_mapping[main_slack_bus_id]
            
            pp.create_ext_grid(
                net,
                bus=pp_bus_idx,
                vm_pu=1.02,  # Slightly higher voltage for power flow
                va_degree=0.0,
                name=f"Slack_{main_slack_bus.name}",
                s_sc_max_mva=10000,  # High short circuit capacity
                s_sc_min_mva=8000,
                rx_max=0.1,
                rx_min=0.1
            )
            
            logger.info(f"Main slack bus: {main_slack_bus.name} at {main_slack_bus.voltage_level}kV")
            
            # For remaining high voltage buses, create generators to represent interconnections
            # But only if they don't already have generators
            for bus_id, bus in slack_buses[1:]:
                if bus.voltage_level >= 345:  # Only for EHV buses
                    pp_bus_idx = self.bus_mapping[bus_id]
                    
                    # Check if this bus already has a generator
                    existing_gens = net.gen[net.gen['bus'] == pp_bus_idx]
                    if len(existing_gens) == 0:
                        pp.create_gen(
                            net,
                            bus=pp_bus_idx,
                            p_mw=0.0,  # Will be determined by power flow
                            vm_pu=1.01,
                            name=f"Interconnection_{bus.name}",
                            max_p_mw=2000,
                            min_p_mw=-2000,  # Can import or export
                            max_q_mvar=1000,
                            min_q_mvar=-1000,
                            controllable=True,
                            type="interconnection"
                        )

def build_pandapower_network(grid: AdvancedGrid, 
                           timestamp: Optional[datetime.datetime] = None,
                           **kwargs) -> Tuple[pp.pandapowerNet, NetworkBuilder]:
    """
    Convenience function to build pandapower network
    
    Args:
        grid: AdvancedGrid topology
        timestamp: Time for load/generation calculations
        **kwargs: Additional parameters for network building
        
    Returns:
        Tuple of (pandapower network, network builder with mappings)
    """
    builder = NetworkBuilder()
    net = builder.build_network(grid, timestamp, **kwargs)
    return net, builder 