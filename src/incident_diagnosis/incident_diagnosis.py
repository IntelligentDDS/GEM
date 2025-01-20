import networkx as nx
import math
import optuna
import numpy as np

DAMPING = 0.1

def get_weight_from_edge_info(topology, clue_tag, statistics = None):
    weight = {}
    nodeList = topology['nodes']
    node2info = {node:{'sum_out': 0, 'max_in': 0, 'max_out': 0} for node in nodeList}
    for edge_key in topology['edge_info']:
        edge = {}
        edge['src'] = edge_key.split('_')[0]
        edge['des'] = edge_key.split('_')[1]
        if statistics is None:
            edge['current'] = np.nan_to_num(topology['edge_info'][edge_key][clue_tag][-1], 0)
            edge['base'] = np.nan_to_num(min(topology['edge_mount_info'][edge_key][clue_tag]), 0)
        else:
            edge['current'] = (np.nan_to_num(topology['edge_info'][edge_key][clue_tag][-1], 0) - statistics[clue_tag+'_mean'])/statistics[clue_tag+'_std'] - (statistics[clue_tag+'_min'] - statistics[clue_tag+'_mean'])/statistics[clue_tag+'_std']
            edge['base'] = (np.nan_to_num(min(topology['edge_mount_info'][edge_key][clue_tag]), 0) - statistics[clue_tag+'_mean'])/statistics[clue_tag+'_std'] - (statistics[clue_tag+'_min'] - statistics[clue_tag+'_mean'])/statistics[clue_tag+'_std']
        edge['deviation'] = edge['current'] - edge['base']
        weight[edge['src']+'-'+edge['des']] = max(edge['deviation'], 0)
        
        node2info[edge['src']]['sum_out'] += max(edge['deviation'], 0)
        node2info[edge['des']]['max_in'] = max(edge['deviation'], node2info[edge['des']]['max_in'])
        node2info[edge['src']]['max_out'] = max(edge['deviation'], node2info[edge['src']]['max_out'])
        
    for node in node2info:
        weight[node+'_self'] = max(node2info[node]['max_in']-node2info[node]['max_out'],0)
        weight[node+'_back_prefix'] = node2info[node]['sum_out']
        
    #print(weight)
    return weight

def get_anomaly_graph(topology, node_clue_tags=[], edge_clue_tags=[], a=None, get_edge_weight=None, edge_backward_factor=0.3):
    anomaly_graph = nx.DiGraph()
    
    nodeList = topology['nodes']

    assert len(node_clue_tags) > 0 or len(edge_clue_tags) > 0, 'At least one clue tag should be provided'

    if a is None:
        #print('No rescale factor provided, using default value')
        a = {}
        for clue_tag in edge_clue_tags:
            a[clue_tag] = 1
        for clue_tag in node_clue_tags:
            a[clue_tag] = 1

    def rescale(x, clue_tag):
        x = np.mean(x)
        return a[clue_tag] * x
    
    for clue_tag in edge_clue_tags:
        edge_weightCal = get_edge_weight(topology, clue_tag)
        
        for edge in topology['edge_info']:
            edgeSrc = edge.split('_')[0]
            edgeDes = edge.split('_')[1]
            if anomaly_graph.has_edge(edgeSrc, edgeDes):
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight = max(rescale(edge_weightCal[edgeSrc + '-' + edgeDes], clue_tag), anomaly_graph.get_edge_data(edgeSrc, edgeDes)['weight']))
            else:
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight = rescale(edge_weightCal[edgeSrc + '-' + edgeDes], clue_tag))

            if anomaly_graph.has_edge(edgeDes, edgeSrc):
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight = max(rescale(edge_backward_factor*(edge_weightCal[edgeSrc+'_back_prefix']-edge_weightCal[edgeSrc + '-' + edgeDes]), clue_tag), anomaly_graph.get_edge_data(edgeDes, edgeSrc)['weight']))
            else:
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight = rescale(edge_backward_factor*(edge_weightCal[edgeSrc+'_back_prefix']-edge_weightCal[edgeSrc + '-' + edgeDes]), clue_tag))
            
        for node in nodeList:
            if anomaly_graph.has_edge(node, node):
                anomaly_graph.add_edge(node, node, weight = max(rescale(edge_weightCal[node+'_self'], clue_tag), anomaly_graph.get_edge_data(node, node)['weight']))
            else:
                anomaly_graph.add_edge(node, node, weight = rescale(edge_weightCal[node+'_self'], clue_tag))

    for clue_tag in node_clue_tags:
        for edge in topology['edge_info']:
            edgeSrc = edge.split('_')[0]
            edgeDes = edge.split('_')[1]
            if anomaly_graph.has_edge(edgeSrc, edgeDes):
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight = max(rescale(topology['node_info'][edgeDes][clue_tag], clue_tag), anomaly_graph.get_edge_data(edgeSrc, edgeDes)['weight']))
            else:
                anomaly_graph.add_edge(edgeSrc, edgeDes, weight = rescale(topology['node_info'][edgeDes][clue_tag], clue_tag))

            if anomaly_graph.has_edge(edgeDes, edgeSrc):
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight = max(rescale(topology['node_info'][edgeSrc][clue_tag], clue_tag), anomaly_graph.get_edge_data(edgeDes, edgeSrc)['weight']))
            else:
                anomaly_graph.add_edge(edgeDes, edgeSrc, weight = rescale(topology['node_info'][edgeSrc][clue_tag], clue_tag))
            
        for node in nodeList:

            if anomaly_graph.has_edge(node, node):
                anomaly_graph.add_edge(node, node, weight = max(rescale(topology['node_info'][node][clue_tag], clue_tag), anomaly_graph.get_edge_data(node, node)['weight']))
            else:
                anomaly_graph.add_edge(node, node, weight = rescale(topology['node_info'][node][clue_tag], clue_tag))

    return anomaly_graph


def root_cause_localization(case, node_clue_tags, edge_clue_tags, a, get_edge_weight=None, edge_backward_factor=0.3):    
    anomalyGraph = get_anomaly_graph(case, node_clue_tags, edge_clue_tags, a, get_edge_weight, edge_backward_factor)
    
    anomaly_score = nx.pagerank(anomalyGraph, alpha=1-DAMPING)#, personalization = personalization)
    anomaly_score_sorted = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

    if len(anomaly_score_sorted) > 0:
        return anomaly_score_sorted[0][0]
    else:
        return 'None'
    

def explain(case, target='root_cause'):
    explanation = []
    explanation_power = dict()
    spatial_exclusion = []
    spatial_exclusion_power = dict()

    explain_target = case[target]
    for node in case['node_info']:
        if explain_target in node:
            explanation.extend(case['node_info'][node].keys())
            for key in case['node_info'][node]:
                if key not in explanation_power:
                    explanation_power[key] = []
                explanation_power[key].append(np.mean(case['node_info'][node][key]))
        else:
            spatial_exclusion.extend(case['node_info'][node].keys())
            for key in case['node_info'][node]:
                if key not in spatial_exclusion_power:
                    spatial_exclusion_power[key] = []
                spatial_exclusion_power[key].append(np.mean(case['node_info'][node][key]))
    
    all_keys = set(explanation) | set(spatial_exclusion)

    for key in explanation_power:
        explanation_power[key] = np.max(explanation_power[key])

    for key in spatial_exclusion_power:
        spatial_exclusion_power[key] = np.max(spatial_exclusion_power[key])

    refined_explanation_power = {key: explanation_power.get(key, 0) - spatial_exclusion_power.get(key, 0) for key in all_keys}

    sorted_refined_explanation_power = sorted(refined_explanation_power.items(), key=lambda x: x[1], reverse=True)

    return sorted_refined_explanation_power


def eval(historical_incident_topologies, node_clue_tags, edge_clue_tags, a, get_edge_weight, edge_backward_factor):
    reward = 0
    for case in historical_incident_topologies:
        pred = root_cause_localization(case, node_clue_tags, edge_clue_tags, a, get_edge_weight, edge_backward_factor)
        if case['root_cause'] in pred:
            reward += 1

    reward /= len(historical_incident_topologies)

    punishment = 0

    for clue_tag in a:
        punishment += np.abs(a[clue_tag])#**2

    return reward, max(punishment, 1)


def optimize(case, node_clue_tags, edge_clue_tags, a, get_edge_weight, edge_backward_factor, historical_incident_topologies, init_clue_tag, range_a=5, num_trials=100):
    #explain
    sorted_refined_explanation_power = explain(case, 'root_cause')

    new_node_clue_tags = node_clue_tags.copy()

    old_a = a.copy()

    if sorted_refined_explanation_power[0][0] not in new_node_clue_tags:
        new_node_clue_tags.append(sorted_refined_explanation_power[0][0])
        old_a[sorted_refined_explanation_power[0][0]] = 0

    def objective(trial):
        new_a = {init_clue_tag: 1}

        for clue_tag in edge_clue_tags:
            new_a[clue_tag] = trial.suggest_float('a:' + clue_tag, 0, range_a)

        for clue_tag in new_node_clue_tags:
            new_a[clue_tag] = trial.suggest_float('a:' + clue_tag, 0, range_a)
        
        reward, punishment = eval(historical_incident_topologies, new_node_clue_tags, edge_clue_tags, new_a, get_edge_weight, edge_backward_factor)

        return reward, punishment
    
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, directions=['maximize', 'minimize'])

    #init trial
    init_a = {init_clue_tag: 1}
    init_params = {'a:' + init_clue_tag: 1}
    init_distributions = {'a:' + init_clue_tag: optuna.distributions.FloatDistribution(0, range_a)}

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

    init_reward_threhold, init_punishment_threhold = eval(historical_incident_topologies, new_node_clue_tags, edge_clue_tags, init_a, get_edge_weight, edge_backward_factor)
    
    init_trial = optuna.trial.create_trial(
            params=init_params,
            distributions=init_distributions,
            values=[init_reward_threhold, init_punishment_threhold],
        )
    #print(len(historical_incident_topologies), init_trial)
    study.add_trial(init_trial)

    # last trial
    old_params = {}
    old_distributions = {}

    for clue_tag in edge_clue_tags:
        old_params['a:' + clue_tag] = old_a[clue_tag]
        old_distributions['a:' + clue_tag] = optuna.distributions.FloatDistribution(0, range_a)

    for clue_tag in new_node_clue_tags:
        old_params['a:' + clue_tag] = old_a[clue_tag]
        old_distributions['a:' + clue_tag] = optuna.distributions.FloatDistribution(0, range_a)

    old_reward_threhold, old_punishment_threhold = eval(historical_incident_topologies, new_node_clue_tags, edge_clue_tags, old_a, get_edge_weight, edge_backward_factor)

    old_trial = optuna.trial.create_trial(
            params=old_params,
            distributions=old_distributions,
            values=[old_reward_threhold, old_punishment_threhold],
        )
    
    study.add_trial(old_trial)

    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    trial_with_highest_reward = max(study.best_trials, key=lambda t: t.values[0])

    print('best trial:', trial_with_highest_reward)

    reward_threhold = max(init_reward_threhold, old_reward_threhold)
    if trial_with_highest_reward.values[0] > reward_threhold:
        print('A better solution found')
        
    for clue_tag in edge_clue_tags:
        a[clue_tag] = trial_with_highest_reward.params['a:' + clue_tag]

    for clue_tag in new_node_clue_tags:
        a[clue_tag] = trial_with_highest_reward.params['a:' + clue_tag]

    print('a:', a)

    node_clue_tags = new_node_clue_tags

    return node_clue_tags, a