"""
Incident Diagnosis Module of Gem

This module implements root cause analysis and incident diagnosis algorithms
of Gem. It uses graph-based approaches to identify the most
likely root causes of incidents based on various clues and metrics.

Key features:
- Graph-based anomaly propagation modeling
- PageRank-based root cause localization
- Multi-clue integration (node and edge features)
- Continual optimization
- Explanation generation for diagnosed incidents
"""

import networkx as nx
import math
import optuna
import numpy as np

# Global damping factor for PageRank algorithm
DAMPING = 0.1


def get_weight_from_edge_info(topology, clue_tag, statistics=None):
    """
    Extract edge weights from topology information based on specified clue.
    
    This function computes edge weights by analyzing the deviation between
    current metric values and baseline values. It also computes node-level
    weights based on incoming and outgoing edge patterns.
    
    Args:
        topology (dict): Topology information containing:
            - 'nodes': List of node identifiers
            - 'edge_info': Dictionary of edge metrics
            - 'edge_mount_info': Dictionary of baseline edge metrics
        clue_tag (str): The specific metric/clue to analyze
        statistics (dict, optional): Normalization statistics for the clue
        
    Returns:
        dict: Dictionary containing computed weights:
            - Edge weights: '{src}-{dst}' -> weight
            - Self weights: '{node}_self' -> weight
            - Backward weights: '{node}_back_prefix' -> weight
    """
    weight = {}
    nodeList = topology['nodes']
    
    # Initialize node information tracking
    node2info = {
        node: {'sum_out': 0, 'max_in': 0, 'max_out': 0} 
        for node in nodeList
    }
    
    # Process each edge in the topology
    for edge_key in topology['edge_info']:
        edge = {}
        edge['src'] = edge_key.split('_')[0]
        edge['des'] = edge_key.split('_')[1]
        
        # Extract current and baseline values
        if statistics is None:
            # Use raw values without normalization
            edge['current'] = np.nan_to_num(topology['edge_info'][edge_key][clue_tag][-1], 0)
            edge['base'] = np.nan_to_num(min(topology['edge_mount_info'][edge_key][clue_tag]), 0)
        else:
            # Apply z-score normalization using provided statistics
            current_raw = np.nan_to_num(topology['edge_info'][edge_key][clue_tag][-1], 0)
            base_raw = np.nan_to_num(min(topology['edge_mount_info'][edge_key][clue_tag]), 0)
            
            # Normalize current value
            edge['current'] = ((current_raw - statistics[clue_tag+'_mean']) / statistics[clue_tag+'_std'] - 
                             (statistics[clue_tag+'_min'] - statistics[clue_tag+'_mean']) / statistics[clue_tag+'_std'])
            
            # Normalize baseline value
            edge['base'] = ((base_raw - statistics[clue_tag+'_mean']) / statistics[clue_tag+'_std'] - 
                          (statistics[clue_tag+'_min'] - statistics[clue_tag+'_mean']) / statistics[clue_tag+'_std'])
        
        # Compute deviation (positive values indicate anomalous increase)
        edge['deviation'] = edge['current'] - edge['base']
        weight[edge['src']+'-'+edge['des']] = max(edge['deviation'], 0)
        
        # Update node-level statistics
        node2info[edge['src']]['sum_out'] += max(edge['deviation'], 0)
        node2info[edge['des']]['max_in'] = max(edge['deviation'], node2info[edge['des']]['max_in'])
        node2info[edge['src']]['max_out'] = max(edge['deviation'], node2info[edge['src']]['max_out'])
    
    # Compute node-level weights
    for node in node2info:
        # Self-loop weight: difference between max incoming and outgoing deviations
        weight[node+'_self'] = max(node2info[node]['max_in'] - node2info[node]['max_out'], 0)
        # Backward propagation weight: sum of outgoing deviations
        weight[node+'_back_prefix'] = node2info[node]['sum_out']
    
    return weight


def get_anomaly_graph(topology, node_clue_tags=[], edge_clue_tags=[], a=None, 
                     get_edge_weight=None, edge_backward_factor=0.3):
    """
    Construct an anomaly propagation graph from topology and clues.
    
    This function builds a directed graph where edge weights represent the
    strength of anomaly propagation between nodes. It integrates multiple
    types of clues (node-based and edge-based) to create a comprehensive
    view of how anomalies might spread through the system.
    
    Args:
        topology (dict): System topology information
        node_clue_tags (list): List of node-based clue identifiers
        edge_clue_tags (list): List of edge-based clue identifiers
        a (dict, optional): Scaling factors for each clue type
        get_edge_weight (function): Function to compute edge weights from clues
        edge_backward_factor (float): Factor for backward edge weight computation
        
    Returns:
        networkx.DiGraph: Directed graph with anomaly propagation weights
    """
    anomaly_graph = nx.DiGraph()
    nodeList = topology['nodes']

    # Ensure at least one type of clue is provided
    assert len(node_clue_tags) > 0 or len(edge_clue_tags) > 0, 'At least one clue tag should be provided'

    # Initialize scaling factors if not provided
    if a is None:
        a = {}
        for clue_tag in edge_clue_tags:
            a[clue_tag] = 1
        for clue_tag in node_clue_tags:
            a[clue_tag] = 1

    def rescale(x, clue_tag):
        """
        Apply scaling factor to clue values.
        
        Args:
            x: Clue value(s)
            clue_tag (str): Clue identifier
            
        Returns:
            float: Scaled clue value
        """
        x = np.mean(x)
        return a[clue_tag] * x
    
    # Process edge-based clues
    for clue_tag in edge_clue_tags:
        edge_weightCal = get_edge_weight(topology, clue_tag)
        
        # Add forward edges
        for edge in topology['edge_info']:
            edgeSrc = edge.split('_')[0]
            edgeDes = edge.split('_')[1]
            
            forward_weight = rescale(edge_weightCal[edgeSrc + '-' + edgeDes], clue_tag)
            
            if anomaly_graph.has_edge(edgeSrc, edgeDes):
                # Update existing edge with maximum weight
                current_weight = anomaly_graph.get_edge_data(edgeSrc, edgeDes)['weight']
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight=max(forward_weight, current_weight))
            else:
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight=forward_weight)

            # Add backward edges (for reverse anomaly propagation)
            backward_weight = rescale(
                edge_backward_factor * (edge_weightCal[edgeSrc+'_back_prefix'] - 
                                      edge_weightCal[edgeSrc + '-' + edgeDes]), 
                clue_tag
            )
            
            if anomaly_graph.has_edge(edgeDes, edgeSrc):
                current_weight = anomaly_graph.get_edge_data(edgeDes, edgeSrc)['weight']
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight=max(backward_weight, current_weight))
            else:
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight=backward_weight)
            
        # Add self-loops for nodes
        for node in nodeList:
            self_weight = rescale(edge_weightCal[node+'_self'], clue_tag)
            
            if anomaly_graph.has_edge(node, node):
                current_weight = anomaly_graph.get_edge_data(node, node)['weight']
                anomaly_graph.add_edge(node, node, weight=max(self_weight, current_weight))
            else:
                anomaly_graph.add_edge(node, node, weight=self_weight)

    # Process node-based clues
    for clue_tag in node_clue_tags:
        # Add edges based on node clue values
        for edge in topology['edge_info']:
            edgeSrc = edge.split('_')[0]
            edgeDes = edge.split('_')[1]
            
            # Forward edge weighted by destination node clue
            forward_weight = rescale(topology['node_info'][edgeDes][clue_tag], clue_tag)
            
            if anomaly_graph.has_edge(edgeSrc, edgeDes):
                current_weight = anomaly_graph.get_edge_data(edgeSrc, edgeDes)['weight']
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight=max(forward_weight, current_weight))
            else:
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight=forward_weight)

            # Backward edge weighted by source node clue
            backward_weight = rescale(topology['node_info'][edgeSrc][clue_tag], clue_tag)
            
            if anomaly_graph.has_edge(edgeDes, edgeSrc):
                current_weight = anomaly_graph.get_edge_data(edgeDes, edgeSrc)['weight']
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight=max(backward_weight, current_weight))
            else:
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight=backward_weight)
            
        # Self-loops weighted by node clue values
        for node in nodeList:
            self_weight = rescale(topology['node_info'][node][clue_tag], clue_tag)
            
            if anomaly_graph.has_edge(node, node):
                current_weight = anomaly_graph.get_edge_data(node, node)['weight']
                anomaly_graph.add_edge(node, node, weight=max(self_weight, current_weight))
            else:
                anomaly_graph.add_edge(node, node, weight=self_weight)

    return anomaly_graph


def root_cause_localization(case, node_clue_tags, edge_clue_tags, a, 
                           get_edge_weight=None, edge_backward_factor=0.3):    
    """
    Identify the most likely root cause using PageRank on the anomaly graph.
    
    This function constructs an anomaly propagation graph and applies PageRank
    algorithm to identify the node most likely to be the root cause of the incident.
    
    Args:
        case (dict): Incident case information
        node_clue_tags (list): Node-based clue identifiers
        edge_clue_tags (list): Edge-based clue identifiers
        a (dict): Scaling factors for clues
        get_edge_weight (function): Edge weight computation function
        edge_backward_factor (float): Backward propagation factor
        
    Returns:
        str: Identifier of the most likely root cause node
    """
    # Build anomaly propagation graph
    anomalyGraph = get_anomaly_graph(
        case, node_clue_tags, edge_clue_tags, a, 
        get_edge_weight, edge_backward_factor
    )
    
    # Apply PageRank to find most influential node
    anomaly_score = nx.pagerank(anomalyGraph, alpha=1-DAMPING)
    anomaly_score_sorted = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

    if len(anomaly_score_sorted) > 0:
        return anomaly_score_sorted[0][0]  # Return top-ranked node
    else:
        return 'None'
    

def explain(case, target='root_cause'):
    """
    Generate explanations for the diagnosed root cause.
    
    This function analyzes the difference in clue patterns between the
    diagnosed root cause area and other areas to provide interpretable
    explanations for why a particular node was identified as the root cause.
    
    Args:
        case (dict): Incident case information
        target (str): Target field to explain (default: 'root_cause')
        
    Returns:
        list: Sorted list of (clue_tag, explanation_power) tuples
    """
    explanation = []
    explanation_power = dict()
    spatial_exclusion = []
    spatial_exclusion_power = dict()

    explain_target = case[target]
    
    # Collect clues from target area and other areas
    for node in case['node_info']:
        if explain_target in node:
            # Clues from the diagnosed root cause area
            explanation.extend(case['node_info'][node].keys())
            for key in case['node_info'][node]:
                if key not in explanation_power:
                    explanation_power[key] = []
                explanation_power[key].append(np.mean(case['node_info'][node][key]))
        else:
            # Clues from other areas (spatial exclusion)
            spatial_exclusion.extend(case['node_info'][node].keys())
            for key in case['node_info'][node]:
                if key not in spatial_exclusion_power:
                    spatial_exclusion_power[key] = []
                spatial_exclusion_power[key].append(np.mean(case['node_info'][node][key]))
    
    all_keys = set(explanation) | set(spatial_exclusion)

    # Compute maximum values for each clue type
    for key in explanation_power:
        explanation_power[key] = np.max(explanation_power[key])

    for key in spatial_exclusion_power:
        spatial_exclusion_power[key] = np.max(spatial_exclusion_power[key])

    # Compute relative explanation power (target area vs. other areas)
    refined_explanation_power = {
        key: explanation_power.get(key, 0) - spatial_exclusion_power.get(key, 0) 
        for key in all_keys
    }

    # Sort by explanation power
    sorted_refined_explanation_power = sorted(
        refined_explanation_power.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_refined_explanation_power


def eval(historical_incident_topologies, node_clue_tags, edge_clue_tags, a, 
         get_edge_weight, edge_backward_factor):
    """
    Evaluate the performance of current clue configuration on historical data.
    
    Args:
        historical_incident_topologies (list): List of historical incident cases
        node_clue_tags (list): Node-based clue identifiers
        edge_clue_tags (list): Edge-based clue identifiers
        a (dict): Scaling factors for clues
        get_edge_weight (function): Edge weight computation function
        edge_backward_factor (float): Backward propagation factor
        
    Returns:
        tuple: (reward, punishment)
            - reward (float): Accuracy on historical cases
            - punishment (float): Regularization penalty
    """
    reward = 0
    
    # Test on each historical case
    for case in historical_incident_topologies:
        pred = root_cause_localization(
            case, node_clue_tags, edge_clue_tags, a, 
            get_edge_weight, edge_backward_factor
        )
        if case['root_cause'] in pred:
            reward += 1

    # Normalize reward by number of cases
    reward /= len(historical_incident_topologies)

    # Compute regularization penalty
    punishment = 0
    for clue_tag in a:
        punishment += np.abs(a[clue_tag])

    return reward, max(punishment, 1)


def optimize(case, node_clue_tags, edge_clue_tags, a, get_edge_weight, 
            edge_backward_factor, historical_incident_topologies, init_clue_tag, 
            range_a=5, num_trials=100):
    """
    Optimize clue weights using Optuna for continual learning.
    
    This function uses the current incident case to identify new relevant clues
    and optimizes the weighting of all clues using historical performance data.
    
    Args:
        case (dict): Current incident case
        node_clue_tags (list): Current node-based clue identifiers
        edge_clue_tags (list): Current edge-based clue identifiers
        a (dict): Current scaling factors
        get_edge_weight (function): Edge weight computation function
        edge_backward_factor (float): Backward propagation factor
        historical_incident_topologies (list): Historical cases for validation
        init_clue_tag (str): Initial clue tag to keep fixed
        range_a (float): Search range for scaling factors
        num_trials (int): Number of optimization trials
        
    Returns:
        tuple: (updated_node_clue_tags, updated_a)
    """
    # Generate explanation for current case to identify new clues
    sorted_refined_explanation_power = explain(case, 'root_cause')

    new_node_clue_tags = node_clue_tags.copy()
    old_a = a.copy()

    # Add most explanatory clue if not already present
    if sorted_refined_explanation_power[0][0] not in new_node_clue_tags:
        new_node_clue_tags.append(sorted_refined_explanation_power[0][0])
        old_a[sorted_refined_explanation_power[0][0]] = 0

    def objective(trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            tuple: (reward, punishment) for multi-objective optimization
        """
        new_a = {init_clue_tag: 1}  # Keep initial clue fixed

        # Suggest values for edge clues
        for clue_tag in edge_clue_tags:
            new_a[clue_tag] = trial.suggest_float('a:' + clue_tag, 0, range_a)

        # Suggest values for node clues
        for clue_tag in new_node_clue_tags:
            new_a[clue_tag] = trial.suggest_float('a:' + clue_tag, 0, range_a)
        
        # Evaluate performance
        reward, punishment = eval(
            historical_incident_topologies, new_node_clue_tags, 
            edge_clue_tags, new_a, get_edge_weight, edge_backward_factor
        )

        return reward, punishment
    
    # Setup for multi-objective optimization
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, directions=['maximize', 'minimize'])

    # Add initial trial with default values
    init_a = {init_clue_tag: 1}
    init_params = {'a:' + init_clue_tag: 1}
    init_distributions = {'a:' + init_clue_tag: optuna.distributions.FloatDistribution(0, range_a)}

    # Setup parameters for all clue types
    for clue_tag in edge_clue_tags:
        if clue_tag != init_clue_tag:
            init_a[clue_tag] = 0
            init_params['a:' + clue_tag] = 0
            init_distributions['a:' + clue_tag] = optuna.distributions.FloatDistribution(0, range_a)

    for clue_tag in new_node_clue_tags:
        if clue_tag != init_clue_tag:
            init_a[clue_tag] = 0
            init_params['a:' + clue_tag] = 0
            init_distributions['a:' + clue_tag] = optuna.distributions.FloatDistribution(0, range_a)

    # Evaluate initial configuration
    init_reward_threshold, init_punishment_threshold = eval(
        historical_incident_topologies, new_node_clue_tags, 
        edge_clue_tags, init_a, get_edge_weight, edge_backward_factor
    )
    
    # Add initial trial to study
    init_trial = optuna.trial.create_trial(
        params=init_params,
        distributions=init_distributions,
        values=[init_reward_threshold, init_punishment_threshold],
    )
    study.add_trial(init_trial)

    # Add previous configuration as a trial
    old_params = {}
    old_distributions = {}

    for clue_tag in edge_clue_tags:
        old_params['a:' + clue_tag] = old_a[clue_tag]
        old_distributions['a:' + clue_tag] = optuna.distributions.FloatDistribution(0, range_a)

    for clue_tag in new_node_clue_tags:
        old_params['a:' + clue_tag] = old_a[clue_tag]
        old_distributions['a:' + clue_tag] = optuna.distributions.FloatDistribution(0, range_a)

    old_reward_threshold, old_punishment_threshold = eval(
        historical_incident_topologies, new_node_clue_tags, 
        edge_clue_tags, old_a, get_edge_weight, edge_backward_factor
    )

    old_trial = optuna.trial.create_trial(
        params=old_params,
        distributions=old_distributions,
        values=[old_reward_threshold, old_punishment_threshold],
    )
    
    study.add_trial(old_trial)

    # Run optimization
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    # Select best trial based on reward
    trial_with_highest_reward = max(study.best_trials, key=lambda t: t.values[0])

    print('best trial:', trial_with_highest_reward)

    # Check if optimization found improvement
    reward_threshold = max(init_reward_threshold, old_reward_threshold)
    if trial_with_highest_reward.values[0] > reward_threshold:
        print('A better solution found')
        
    # Update scaling factors with optimized values
    for clue_tag in edge_clue_tags:
        a[clue_tag] = trial_with_highest_reward.params['a:' + clue_tag]

    for clue_tag in new_node_clue_tags:
        a[clue_tag] = trial_with_highest_reward.params['a:' + clue_tag]

    print('a:', a)

    node_clue_tags = new_node_clue_tags

    return node_clue_tags, a