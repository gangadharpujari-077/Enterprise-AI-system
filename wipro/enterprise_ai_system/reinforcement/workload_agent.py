"""
Reinforcement Learning Workload Management Agent
Simple Q-learning agent for intelligent workload decisions
"""

import numpy as np
from typing import Tuple, Dict, List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkloadManagementAgent:
    """Simple Q-learning agent for workload optimization"""
    
    # Define states and actions
    STATES = {
        'low_workload_low_risk': 0,
        'medium_workload_medium_risk': 1,
        'high_workload_high_risk': 2,
        'low_workload_high_risk': 3,
        'high_workload_low_risk': 4
    }
    
    ACTIONS = {
        'maintain_workload': 0,
        'reduce_workload': 1,
        'reassign_task': 2,
        'increase_deadline': 3,
        'prioritize_urgent': 4
    }
    
    def __init__(self, learning_rate: float = 0.1, 
                 discount_factor: float = 0.99,
                 exploration_rate: float = 0.1):
        """
        Initialize Q-learning agent
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Epsilon for epsilon-greedy policy
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Initialize Q-table
        n_states = len(self.STATES)
        n_actions = len(self.ACTIONS)
        self.q_table = np.zeros((n_states, n_actions))
        
        # Track episode rewards
        self.episode_rewards = []
        self.training_episodes = 0
    
    def get_state(self, workload_score: float, 
                  delay_risk: float, 
                  burnout_risk: float) -> int:
        """
        Map employee metrics to discrete state
        
        Args:
            workload_score: Current workload (0-1)
            delay_risk: Delay risk probability (0-1)
            burnout_risk: Burnout risk probability (0-1)
        
        Returns:
            State index
        """
        high_workload = workload_score > 0.6
        low_workload = workload_score < 0.4
        high_risk = (delay_risk + burnout_risk) / 2 > 0.5
        low_risk = (delay_risk + burnout_risk) / 2 < 0.3
        
        if low_workload and low_risk:
            return self.STATES['low_workload_low_risk']
        elif high_workload and high_risk:
            return self.STATES['high_workload_high_risk']
        elif not low_workload and not high_workload and not low_risk and not high_risk:
            return self.STATES['medium_workload_medium_risk']
        elif low_workload and high_risk:
            return self.STATES['low_workload_high_risk']
        elif high_workload and low_risk:
            return self.STATES['high_workload_low_risk']
        else:
            return self.STATES['medium_workload_medium_risk']
    
    def select_action(self, state: int, use_exploration: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            use_exploration: Whether to use exploration (for training)
        
        Returns:
            Action index
        """
        if use_exploration and np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.choice(list(self.ACTIONS.values()))
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state, :])
    
    def calculate_reward(self, previous_state: int,
                        action: int,
                        new_workload: float,
                        new_delay_risk: float,
                        new_burnout_risk: float) -> float:
        """
        Calculate reward for action
        
        Args:
            previous_state: Previous state
            action: Action taken
            new_workload: New workload after action
            new_delay_risk: New delay risk after action
            new_burnout_risk: New burnout risk after action
        
        Returns:
            Reward value
        """
        # Base reward: minimize risk
        risk_reduction = (new_delay_risk + new_burnout_risk) / 2
        reward = -risk_reduction * 10
        
        # Workload balance reward (between 0.3 and 0.7 is ideal)
        if 0.3 < new_workload < 0.7:
            reward += 5
        else:
            reward -= 5
        
        # Action-specific rewards
        if action == self.ACTIONS['reduce_workload']:
            if new_burnout_risk < 0.4:
                reward += 10
        elif action == self.ACTIONS['maintain_workload']:
            if new_workload > 0.4 and new_delay_risk < 0.3:
                reward += 8
        elif action == self.ACTIONS['reassign_task']:
            if new_delay_risk < 0.3:
                reward += 10
        
        return reward
    
    def update_q_value(self, state: int, 
                      action: int,
                      reward: float,
                      next_state: int):
        """
        Update Q-value using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        
        # Q-learning update: Q(s,a) = Q(s,a) + α(r + γ*max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
    
    def train(self, episodes: int = 100,
             trajectories: List[Dict] = None):
        """
        Train the agent
        
        Args:
            episodes: Number of training episodes
            trajectories: Optional list of historical trajectories
        """
        if trajectories is None:
            trajectories = []
        
        for episode in range(episodes):
            episode_reward = 0
            
            # Simulate episode or use historical trajectory
            for step in range(10):  # 10 steps per episode
                # Generate random state and action or use trajectory
                if trajectories and step < len(trajectories):
                    state = self.get_state(
                        trajectories[step].get('workload', 0.5),
                        trajectories[step].get('delay_risk', 0.3),
                        trajectories[step].get('burnout_risk', 0.3)
                    )
                else:
                    state = np.random.choice(list(self.STATES.values()))
                
                # Select action with exploration
                action = self.select_action(state, use_exploration=True)
                
                # Simulate next state and reward
                new_workload = np.random.random()
                new_delay_risk = np.random.random()
                new_burnout_risk = np.random.random()
                next_state = self.get_state(new_workload, new_delay_risk, new_burnout_risk)
                
                # Calculate reward
                reward = self.calculate_reward(state, action, new_workload, 
                                             new_delay_risk, new_burnout_risk)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            self.training_episodes += 1
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    def recommend_action(self, workload_score: float,
                        delay_risk: float,
                        burnout_risk: float) -> Dict:
        """
        Recommend action for given employee state
        
        Args:
            workload_score: Current workload
            delay_risk: Delay risk probability
            burnout_risk: Burnout risk probability
        
        Returns:
            Dictionary with recommended action and explanation
        """
        state = self.get_state(workload_score, delay_risk, burnout_risk)
        action_idx = self.select_action(state, use_exploration=False)
        
        # Get action name
        action_name = [k for k, v in self.ACTIONS.items() if v == action_idx][0]
        
        # Get Q-values for this state
        q_values = self.q_table[state, :]
        
        return {
            'state': [k for k, v in self.STATES.items() if v == state][0],
            'recommended_action': action_name,
            'confidence': float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-8)),
            'current_risk': (delay_risk + burnout_risk) / 2,
            'current_workload': workload_score,
            'explanation': self._get_action_explanation(action_name)
        }
    
    @staticmethod
    def _get_action_explanation(action: str) -> str:
        """Get explanation for action"""
        explanations = {
            'maintain_workload': 'Employee is performing well. Maintain current workload.',
            'reduce_workload': 'High risk detected. Reduce workload immediately.',
            'reassign_task': 'Consider reassigning non-critical tasks to other team members.',
            'increase_deadline': 'Extend deadlines to reduce pressure.',
            'prioritize_urgent': 'Focus on most critical tasks only.'
        }
        return explanations.get(action, 'Unknown action')
    
    def get_q_table_summary(self) -> Dict:
        """Get summary of Q-table"""
        return {
            'states': self.STATES,
            'actions': self.ACTIONS,
            'average_q_values': {
                f"state_{k}": float(np.mean(self.q_table[v, :]))
                for k, v in self.STATES.items()
            },
            'training_episodes': self.training_episodes
        }
    
    def save_agent(self, path: str):
        """Save agent state"""
        state_dict = {
            'q_table': self.q_table.tolist(),
            'episode_rewards': self.episode_rewards,
            'training_episodes': self.training_episodes
        }
        with open(path, 'w') as f:
            json.dump(state_dict, f)
        logger.info(f"Agent saved to {path}")
    
    def load_agent(self, path: str):
        """Load agent state"""
        with open(path, 'r') as f:
            state_dict = json.load(f)
        
        self.q_table = np.array(state_dict['q_table'])
        self.episode_rewards = state_dict['episode_rewards']
        self.training_episodes = state_dict['training_episodes']
        logger.info(f"Agent loaded from {path}")
