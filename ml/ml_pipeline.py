"""
Machine Learning Pipeline for Power System Mitigation

Implements the progression: Random Forest -> Graph Neural Network -> Reinforcement Learning
for learning optimal mitigation strategies from simulation data.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML frameworks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# Reinforcement learning
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Stable-baselines3 not available. RL components will be disabled.")

import gymnasium as gym
from gymnasium import spaces

from loguru import logger
from grid.advanced_grid import AdvancedGrid
from simulation.contingency_analyzer import ContingencyResult
from simulation.mitigation_engine import MitigationPlan, ExecutionResult


@dataclass
class MLPipelineConfig:
    """Configuration for ML pipeline"""
    # Data paths
    simulation_data_path: str = "outputs/simulation_dataset.csv"
    contingency_results_path: str = "outputs/contingency_results.csv"
    mitigation_plans_path: str = "outputs/mitigation_plans/"
    
    # Model paths
    model_save_path: str = "models/"
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cross_val_folds: int = 5
    
    # RF parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    
    # GNN parameters
    gnn_hidden_dim: int = 64
    gnn_learning_rate: float = 0.001
    gnn_epochs: int = 100
    gnn_batch_size: int = 32
    
    # RL parameters
    rl_total_timesteps: int = 100000
    rl_learning_rate: float = 0.0003


class PowerSystemFeatureExtractor:
    """Extract and engineer features for ML models"""
    
    def __init__(self, grid: AdvancedGrid):
        self.grid = grid
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_tabular_features(self, contingency_results: List[ContingencyResult]) -> pd.DataFrame:
        """Extract tabular features from contingency results"""
        
        features = []
        
        for result in contingency_results:
            # Basic contingency features
            feature_dict = {
                'contingency_type': result.contingency.type,
                'num_elements': len(result.contingency.elements),
                'probability': result.contingency.probability,
                'converged': int(result.converged),
                'solve_time': result.solve_time,
                'iterations': result.iterations,
                
                # System state features
                'total_generation_mw': result.total_generation_mw,
                'total_load_mw': result.total_load_mw,
                'total_losses_mw': result.total_losses_mw,
                'loss_percentage': result.total_losses_mw / max(result.total_load_mw, 1) * 100,
                'generation_margin': (result.total_generation_mw - result.total_load_mw) / max(result.total_load_mw, 1),
                
                # Voltage features
                'max_voltage_pu': result.max_voltage_pu,
                'min_voltage_pu': result.min_voltage_pu,
                'voltage_range_pu': result.max_voltage_pu - result.min_voltage_pu,
                'voltage_violations': len([v for v in result.post_violations if 'voltage' in v.violation_type]),
                
                # Thermal features
                'max_line_loading_pct': result.max_line_loading_pct,
                'thermal_violations': len([v for v in result.post_violations if 'thermal' in v.violation_type]),
                
                # Violation features
                'total_violations': len(result.post_violations),
                'new_violations': len(result.new_violations),
                'critical_violations': len([v for v in result.post_violations if v.critical]),
                'severity_score': result.severity_score,
                
                # Topology features
                'islands_created': result.islands_created,
                'islanded_buses_count': len(result.islanded_buses)
            }
            
            # Add element-specific features
            for element_id in result.contingency.elements:
                if element_id in self.grid.generators:
                    gen = self.grid.generators[element_id]
                    feature_dict.update({
                        'outaged_gen_capacity_mw': gen.capacity_mw,
                        'outaged_gen_type': gen.type,
                        'outaged_gen_is_baseload': int(gen.is_baseload)
                    })
                elif element_id in self.grid.lines:
                    line = self.grid.lines[element_id]
                    feature_dict.update({
                        'outaged_line_capacity_mw': getattr(line, 'capacity_mw', 200),
                        'outaged_line_voltage_kv': getattr(line, 'voltage_kv', 115),
                        'outaged_line_length_km': getattr(line, 'length_km', 50)
                    })
            
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        
        # Fill missing values for elements not present
        df['outaged_gen_capacity_mw'] = df['outaged_gen_capacity_mw'].fillna(0)
        df['outaged_gen_type'] = df['outaged_gen_type'].fillna('none')
        df['outaged_gen_is_baseload'] = df['outaged_gen_is_baseload'].fillna(0)
        df['outaged_line_capacity_mw'] = df['outaged_line_capacity_mw'].fillna(0)
        df['outaged_line_voltage_kv'] = df['outaged_line_voltage_kv'].fillna(0)
        df['outaged_line_length_km'] = df['outaged_line_length_km'].fillna(0)
        
        return df
    
    def extract_graph_features(self, contingency_results: List[ContingencyResult]) -> List[Data]:
        """Extract graph-based features for GNN"""
        
        # Build base topology graph
        edge_index, edge_attr = self._build_topology_graph()
        
        graph_data = []
        
        for result in contingency_results:
            # Node features (per bus)
            node_features = self._extract_bus_features(result)
            
            # Modify graph for contingency (remove outaged elements)
            cont_edge_index, cont_edge_attr = self._apply_contingency_to_graph(
                edge_index, edge_attr, result.contingency.elements
            )
            
            # Create PyTorch Geometric data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(cont_edge_index, dtype=torch.long),
                edge_attr=torch.tensor(cont_edge_attr, dtype=torch.float),
                y=torch.tensor([result.severity_score], dtype=torch.float)
            )
            
            graph_data.append(data)
        
        return graph_data
    
    def _build_topology_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build adjacency matrix and edge features from grid topology"""
        
        # Create bus index mapping
        bus_ids = sorted(self.grid.buses.keys())
        bus_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        
        edges = []
        edge_features = []
        
        # Add edges from transmission lines
        for line_id, line in self.grid.lines.items():
            from_idx = bus_to_idx.get(line.from_bus)
            to_idx = bus_to_idx.get(line.to_bus)
            
            if from_idx is not None and to_idx is not None:
                # Add both directions (undirected graph)
                edges.extend([[from_idx, to_idx], [to_idx, from_idx]])
                
                # Line features: capacity, impedance, length
                line_features = [
                    getattr(line, 'capacity_mw', 200),
                    getattr(line, 'voltage_kv', 115),
                    getattr(line, 'length_km', 50)
                ]
                edge_features.extend([line_features, line_features])
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0))
        edge_attr = np.array(edge_features) if edge_features else np.zeros((0, 3))
        
        return edge_index, edge_attr
    
    def _extract_bus_features(self, result: ContingencyResult) -> np.ndarray:
        """Extract features for each bus"""
        
        bus_ids = sorted(self.grid.buses.keys())
        features = []
        
        for bus_id in bus_ids:
            bus = self.grid.buses[bus_id]
            
            # Bus-level features
            bus_features = [
                getattr(bus, 'voltage_level', 115),  # Voltage level
                len(getattr(bus, 'generators', [])),  # Number of generators
                sum([self.grid.generators[g].capacity_mw for g in getattr(bus, 'generators', [])]),  # Generation capacity
                getattr(bus, 'total_load_mw', 10),  # Load
            ]
            
            # Violation features for this bus
            bus_violations = [v for v in result.post_violations if v.location == bus_id]
            bus_features.extend([
                len(bus_violations),
                max([v.severity for v in bus_violations], default=0),
                int(any(v.critical for v in bus_violations))
            ])
            
            features.append(bus_features)
        
        return np.array(features, dtype=np.float32)
    
    def _apply_contingency_to_graph(self, edge_index: np.ndarray, edge_attr: np.ndarray, 
                                  outaged_elements: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outaged elements from graph"""
        
        # For simplicity, return original graph
        # In full implementation, would remove edges corresponding to outaged lines
        return edge_index, edge_attr


class RandomForestPipeline:
    """Random Forest classifier/regressor for RTCA recommendations"""
    
    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.classification_model = None
        self.regression_model = None
        self.feature_extractor = None
        self.preprocessor = None
        
    def prepare_data(self, contingency_results: List[ContingencyResult],
                    mitigation_plans: List[MitigationPlan]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data for RF models"""
        
        # Extract features
        features_df = self.feature_extractor.extract_tabular_features(contingency_results)
        
        # Create labels from mitigation plans
        mitigation_types = []
        mitigation_amounts = []
        
        for result in contingency_results:
            # Find corresponding mitigation plan
            matching_plans = [p for p in mitigation_plans if p.contingency_id == result.contingency.id]
            
            if matching_plans and matching_plans[0].action_sequence:
                # Use first action type as primary mitigation
                first_action = matching_plans[0].action_sequence[0]
                mitigation_types.append(first_action.type)
                
                # Extract amount based on action type
                if first_action.type == 'redispatch':
                    mitigation_amounts.append(first_action.parameters.get('delta_mw', 0))
                elif first_action.type == 'load_shed':
                    mitigation_amounts.append(first_action.parameters.get('shed_mw', 0))
                else:
                    mitigation_amounts.append(0)
            else:
                mitigation_types.append('none')
                mitigation_amounts.append(0)
        
        classification_labels = pd.Series(mitigation_types)
        regression_labels = pd.Series(mitigation_amounts)
        
        return features_df, classification_labels, regression_labels
    
    def train(self, features_df: pd.DataFrame, classification_labels: pd.Series, 
             regression_labels: pd.Series):
        """Train Random Forest models"""
        
        logger.info("Training Random Forest models...")
        
        # Define preprocessor
        categorical_features = ['contingency_type', 'outaged_gen_type']
        numerical_features = [col for col in features_df.columns if col not in categorical_features]
        
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Classification pipeline
        self.classification_model = Pipeline([
            ('prep', self.preprocessor),
            ('clf', RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state
            ))
        ])
        
        # Regression pipeline  
        self.regression_model = Pipeline([
            ('prep', self.preprocessor),
            ('reg', RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state
            ))
        ])
        
        # Train models
        self.classification_model.fit(features_df, classification_labels)
        self.regression_model.fit(features_df, regression_labels)
        
        # Cross-validation evaluation
        cv_scores_clf = cross_val_score(self.classification_model, features_df, classification_labels, 
                                       cv=self.config.cross_val_folds)
        cv_scores_reg = cross_val_score(self.regression_model, features_df, regression_labels, 
                                       cv=self.config.cross_val_folds)
        
        logger.info(f"Classification CV Score: {cv_scores_clf.mean():.3f} ± {cv_scores_clf.std():.3f}")
        logger.info(f"Regression CV Score: {cv_scores_reg.mean():.3f} ± {cv_scores_reg.std():.3f}")
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained models"""
        
        mitigation_types = self.classification_model.predict(features_df)
        mitigation_amounts = self.regression_model.predict(features_df)
        
        return mitigation_types, mitigation_amounts
    
    def save_models(self):
        """Save trained models"""
        model_path = Path(self.config.model_save_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        with open(model_path / "rf_classification.pkl", 'wb') as f:
            pickle.dump(self.classification_model, f)
        
        with open(model_path / "rf_regression.pkl", 'wb') as f:
            pickle.dump(self.regression_model, f)
        
        logger.info("Random Forest models saved")


class RTCA_GNN(nn.Module):
    """Graph Neural Network for topology-aware RTCA"""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_actions)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final prediction
        return self.lin(x)


class GNNPipeline:
    """Graph Neural Network pipeline for topology-aware recommendations"""
    
    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, graph_data: List[Data], labels: List[str]):
        """Train GNN model"""
        
        logger.info("Training Graph Neural Network...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        # Update labels in graph data
        for i, data in enumerate(graph_data):
            data.y = torch.tensor([encoded_labels[i]], dtype=torch.long)
        
        # Split data
        train_size = int(0.8 * len(graph_data))
        train_data = graph_data[:train_size]
        val_data = graph_data[train_size:]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.config.gnn_batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.gnn_batch_size)
        
        # Initialize model
        node_features = train_data[0].x.shape[1] if train_data else 7
        self.model = RTCA_GNN(node_features, self.config.gnn_hidden_dim, num_classes)
        self.model = self.model.to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.gnn_learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config.gnn_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            if epoch % 10 == 0:
                val_acc = self._evaluate(val_loader)
                logger.info(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc:.3f}")
        
        logger.info("GNN training completed")
    
    def _evaluate(self, data_loader):
        """Evaluate model on validation data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        return correct / total if total > 0 else 0


class GridSimEnv(gym.Env):
    """RL Environment for power grid mitigation"""
    
    def __init__(self, grid: AdvancedGrid, contingency_results: List[ContingencyResult]):
        super().__init__()
        
        self.grid = grid
        self.contingency_results = contingency_results
        self.current_result = None
        self.step_count = 0
        self.max_steps = 5
        
        # Action space: discrete actions for different mitigation strategies
        self.action_space = spaces.Discrete(6)  # redispatch, load_shed, reactive, tap, topology, none
        
        # Observation space: system state features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        self.current_result = np.random.choice(self.contingency_results)
        self.step_count = 0
        
        state = self._get_state()
        info = {'contingency_id': self.current_result.contingency.id}
        
        return state, info
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        
        # Execute mitigation action
        reward = self._calculate_reward(action)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or self._violations_cleared()
        
        state = self._get_state()
        info = {'action_taken': action, 'violations_remaining': len(self.current_result.new_violations)}
        
        return state, reward, done, False, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        
        if self.current_result is None:
            return np.zeros(20, dtype=np.float32)
        
        state = [
            self.current_result.total_generation_mw / 10000,  # Normalize
            self.current_result.total_load_mw / 10000,
            self.current_result.max_voltage_pu,
            self.current_result.min_voltage_pu,
            self.current_result.max_line_loading_pct / 100,
            len(self.current_result.new_violations),
            len([v for v in self.current_result.new_violations if v.critical]),
            self.current_result.severity_score / 10,
            self.current_result.islands_created,
            len(self.current_result.contingency.elements),
            # Add more state features...
        ]
        
        # Pad to fixed size
        state.extend([0.0] * (20 - len(state)))
        
        return np.array(state[:20], dtype=np.float32)
    
    def _calculate_reward(self, action) -> float:
        """Calculate reward for taking an action"""
        
        # Base penalty for taking any action
        reward = -1.0
        
        # Large penalty for remaining violations
        reward -= len(self.current_result.new_violations) * 10.0
        
        # Penalty for critical violations
        critical_violations = len([v for v in self.current_result.new_violations if v.critical])
        reward -= critical_violations * 50.0
        
        # Reward for appropriate action selection
        if action == 0 and any('thermal' in v.violation_type for v in self.current_result.new_violations):
            reward += 5.0  # Redispatch for thermal issues
        elif action == 1 and critical_violations > 0:
            reward += 10.0  # Load shed for critical situations
        
        return reward
    
    def _violations_cleared(self) -> bool:
        """Check if all violations are cleared"""
        return len(self.current_result.new_violations) == 0


class RLPipeline:
    """Reinforcement Learning pipeline for sequential mitigation planning"""
    
    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.env = None
        self.model = None
        
    def train(self, grid: AdvancedGrid, contingency_results: List[ContingencyResult]):
        """Train RL agent"""
        
        if not RL_AVAILABLE:
            logger.warning("RL training skipped - stable-baselines3 not available")
            return
        
        logger.info("Training RL agent...")
        
        # Create environment
        self.env = GridSimEnv(grid, contingency_results)
        
        # Create model
        self.model = PPO("MlpPolicy", self.env, 
                        learning_rate=self.config.rl_learning_rate,
                        verbose=1)
        
        # Train
        self.model.learn(total_timesteps=self.config.rl_total_timesteps)
        
        logger.info("RL training completed")
    
    def save_model(self):
        """Save trained RL model"""
        if self.model and RL_AVAILABLE:
            model_path = Path(self.config.model_save_path)
            model_path.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path / "ppo_grid_mitigation")


class MLPipeline:
    """Main ML pipeline orchestrator"""
    
    def __init__(self, grid: AdvancedGrid, config: Optional[MLPipelineConfig] = None):
        self.grid = grid
        self.config = config or MLPipelineConfig()
        
        self.feature_extractor = PowerSystemFeatureExtractor(grid)
        self.rf_pipeline = RandomForestPipeline(self.config)
        self.gnn_pipeline = GNNPipeline(self.config)
        self.rl_pipeline = RLPipeline(self.config)
        
    def run_full_pipeline(self, contingency_results: List[ContingencyResult],
                         mitigation_plans: List[MitigationPlan]):
        """Run complete ML pipeline: RF -> GNN -> RL"""
        
        logger.info("Starting ML pipeline...")
        
        # Phase 1: Random Forest
        logger.info("Phase 1: Training Random Forest models")
        self.rf_pipeline.feature_extractor = self.feature_extractor
        
        features_df, class_labels, reg_labels = self.rf_pipeline.prepare_data(
            contingency_results, mitigation_plans
        )
        
        self.rf_pipeline.train(features_df, class_labels, reg_labels)
        self.rf_pipeline.save_models()
        
        # Phase 2: Graph Neural Network
        logger.info("Phase 2: Training Graph Neural Network")
        graph_data = self.feature_extractor.extract_graph_features(contingency_results)
        mitigation_types = [plan.action_sequence[0].type if plan.action_sequence else 'none' 
                           for plan in mitigation_plans]
        
        self.gnn_pipeline.train(graph_data, mitigation_types)
        
        # Phase 3: Reinforcement Learning
        logger.info("Phase 3: Training RL agent")
        self.rl_pipeline.train(self.grid, contingency_results)
        self.rl_pipeline.save_model()
        
        logger.info("ML pipeline completed successfully")
        
        return {
            'rf_trained': True,
            'gnn_trained': True,
            'rl_trained': RL_AVAILABLE,
            'total_scenarios': len(contingency_results)
        }


if __name__ == "__main__":
    # Example usage
    from grid.advanced_grid import AdvancedGrid
    from simulation.contingency_analyzer import ContingencyAnalyzer
    from simulation.mitigation_engine import MitigationEngine
    from simulation.contingency_generator import load_contingencies
    from simulation.build_network import build_pandapower_network
    
    # Create test setup
    regions = ['A', 'B', 'C']
    grid = AdvancedGrid(regions, buses_per_region=100)
    
    # Load data (assuming it exists)
    contingencies = load_contingencies("data/contingencies")[:20]  # Test subset
    
    if contingencies:
        # Analyze contingencies
        base_net, _ = build_pandapower_network(grid)
        analyzer = ContingencyAnalyzer(grid)
        results = analyzer.analyze_all_contingencies(base_net, contingencies)
        
        # Generate mitigation plans
        engine = MitigationEngine(grid)
        plans = []
        for result in results:
            if result.new_violations:
                plan = engine.generate_mitigation_plan(result, base_net)
                if plan:
                    plans.append(plan)
        
        # Run ML pipeline
        if plans:
            ml_pipeline = MLPipeline(grid)
            pipeline_results = ml_pipeline.run_full_pipeline(results, plans)
            print(f"ML Pipeline Results: {pipeline_results}")
        else:
            print("No mitigation plans generated for ML training")
    else:
        print("No contingencies found. Generate contingencies first.") 