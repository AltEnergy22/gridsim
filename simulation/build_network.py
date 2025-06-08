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
import math
import random

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
        self._add_voltage_control_devices(net, grid)
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
                       timestamp: Optional[datetime.datetime] = None):
        """Add generators with proper voltage control and economic dispatch"""
        logger.info("Adding generators with voltage control...")
        
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Implement economic dispatch for better voltage control
        generators_data = []
        for gen_id, generator in grid.generators.items():
            gen_data = {
                'generator': generator,
                'marginal_cost': generator.marginal_cost(),
                'available_capacity': generator.available_capacity(timestamp)
            }
            generators_data.append(gen_data)
        
        # Sort by marginal cost for economic dispatch
        generators_data.sort(key=lambda x: x['marginal_cost'])
        
        # Calculate total load for proper dispatch
        total_load = sum(bus.load.total_load(timestamp, 25, 1.0, False) 
                        for bus in grid.buses.values())
        
        # Target generation including losses and reserves
        target_generation = total_load * 1.05  # 5% for losses and reserves
        
        # Dispatch generators economically
        cumulative_dispatch = 0
        
        for i, gen_data in enumerate(generators_data):
            generator = gen_data['generator']
            available = gen_data['available_capacity']
            
            # Determine if this generator should be voltage controlling
            # - Large baseload units at transmission level control voltage
            # - Swing bus (first/largest generator) definitely controls voltage
            is_voltage_controlling = (
                i == 0 or  # Swing/largest generator
                (generator.type in ['nuclear', 'coal'] and 
                 generator.capacity_mw > 200 and
                 cumulative_dispatch < target_generation * 0.8)  # Major baseload units
            )
            
            # Calculate dispatch for this generator
            remaining_need = max(0, target_generation - cumulative_dispatch)
            
            if remaining_need > 0:
                # Dispatch between minimum and available capacity
                min_output = available * 0.3 if generator.is_baseload else 0
                max_output = available * 0.95  # Leave some margin
                
                dispatch = min(max_output, max(min_output, remaining_need))
            else:
                dispatch = 0
                
            cumulative_dispatch += dispatch
            
            # Get the bus index for this generator
            bus_idx = self.bus_mapping.get(generator.bus_id)
            if bus_idx is None:
                logger.warning(f"Bus {generator.bus_id} not found for generator {gen_id}")
                continue
            
            # Calculate reactive power limits based on capacity
            max_q = generator.capacity_mw * 0.5  # Typical Q capability
            min_q = -generator.capacity_mw * 0.3
            
            if is_voltage_controlling:
                # PV bus - controls voltage, Q is variable within limits
                target_voltage = self._get_target_voltage(generator, grid.buses[generator.bus_id])
                
                pp.create_gen(net,
                            bus=bus_idx,
                            p_mw=dispatch,
                            vm_pu=target_voltage,  # Voltage setpoint
                            name=generator.name,
                            max_p_mw=available,
                            min_p_mw=0,
                            max_q_mvar=max_q,
                            min_q_mvar=min_q,
                            controllable=True,
                            type=generator.type)
            else:
                # PQ bus - fixed P and Q
                # Estimate Q based on power factor
                power_factor = 0.9  # Typical for thermal units
                q_dispatch = dispatch * math.tan(math.acos(power_factor))
                q_dispatch = max(min_q, min(max_q, q_dispatch))
                
                pp.create_gen(net,
                            bus=bus_idx,
                            p_mw=dispatch,
                            q_mvar=q_dispatch,
                            name=generator.name,
                            max_p_mw=available,
                            min_p_mw=0,
                            max_q_mvar=max_q,
                            min_q_mvar=min_q,
                            controllable=True,
                            type=generator.type)
        
        logger.info(f"Added {len(grid.generators)} generators with economic dispatch")
        logger.info(f"Total dispatched: {cumulative_dispatch:.1f} MW vs target: {target_generation:.1f} MW")
    
    def _get_target_voltage(self, generator: 'Generator', bus: 'Bus') -> float:
        """Determine conservative target voltage setpoint for voltage-controlling generators"""
        # More conservative voltage setpoints to prevent cascading overvoltages
        voltage_level = bus.voltage_level
        
        if voltage_level >= 500:  # EHV transmission (500kV+)
            return 1.00   # Nominal voltage for highest levels
        elif voltage_level >= 345:  # EHV transmission (345kV)
            return 1.005  # Very slight boost
        elif voltage_level >= 138:  # HV transmission (138-230kV) 
            return 1.000  # Nominal voltage
        elif voltage_level >= 69:   # Sub-transmission
            return 0.995  # Slightly below nominal
        else:  # Distribution
            return 1.000
            
    def _add_voltage_control_devices(self, net: pp.pandapowerNet, grid: AdvancedGrid):
        """Add voltage control devices like capacitor banks and reactors"""
        logger.info("Adding voltage control devices...")
        
        # Add shunt capacitors at strategic locations
        for bus_id, bus in grid.buses.items():
            bus_idx = self.bus_mapping.get(bus_id)
            if bus_idx is None:
                continue
                
            # Add capacitors at transmission substations with high load
            if (bus.voltage_level >= 138 and 
                bus.substation_id and 
                random.random() < 0.3):  # 30% of transmission buses get capacitors
                
                # Size capacitor based on local load
                local_load = bus.load.total_load(datetime.datetime.now(), 25, 1.0, False)
                cap_size = local_load * 0.3  # 30% of load as reactive support
                
                pp.create_shunt(net,
                              bus=bus_idx,
                              q_mvar=cap_size,
                              name=f"CAP_{bus_id}",
                              controllable=True)
    
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
        """Add transformers with voltage control"""
        logger.info("Adding transformers with voltage control...")
        
        for trafo_id, trafo in grid.transformers.items():
            hv_bus = self.bus_mapping[trafo.from_bus]
            lv_bus = self.bus_mapping[trafo.to_bus]
            
            # Get voltage levels
            hv_kv = self._get_base_voltage(trafo.from_level)
            lv_kv = self._get_base_voltage(trafo.to_level)
            
            # Calculate transformer parameters
            vk_percent = trafo.impedance_pu * 100  # Convert to percentage
            vkr_percent = vk_percent * 0.1  # Assume R is 10% of total impedance
            
            # Determine if this transformer should have voltage control
            should_control_voltage = (
                trafo.capacity_mva >= 100 and  # Large transformers
                trafo.from_level >= 138 and    # From transmission level
                trafo.to_level <= trafo.from_level / 2  # Significant step down
            )
            
            if should_control_voltage:
                # Enhanced transformer with voltage control
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
                    pfe_kw=trafo.capacity_mva * 1.0,
                    i0_percent=0.5,
                    tap_side="hv",
                    tap_neutral=0,
                    tap_min=-16,  # ±20% voltage control range  
                    tap_max=16,
                    tap_step_percent=1.25,
                    tap_pos=0,
                    max_loading_percent=120.0  # Allow some overload
                )
            else:
                # Standard transformer
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
                    pfe_kw=trafo.capacity_mva * 1.0,
                    i0_percent=0.5,
                    tap_side="hv",
                    tap_neutral=0,
                    tap_min=-10,
                    tap_max=10,
                    tap_step_percent=1.25,
                    tap_pos=0,
                    max_loading_percent=100.0
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
                vm_pu=1.00,  # Nominal voltage to prevent voltage escalation
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