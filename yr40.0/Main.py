#!/usr/bin/env python
# Degrade and fire

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import math
from pathlib import Path
from numpy import random


import importlib as imp


imp.reload(Classy)

''' main Distributed Delay Stochastic Simulation Algorithm 
 http://localhost:8888/edit/BioResearch/python/Functions.py#   NOTE: There is no checking for negative values in this version.'''
#def InitProteins(alpha, beta, yr, R0, C0, tau):
#    return(alpha-yr)(tau-c0*(sqrt(alpha/yr)-1)/yr)

def gillespie(reactions_list, stop_time, initial_state_vector):
    intitilized = initialize(initial_state_vector)
    state_vector = intitilized[0]
    current_time = intitilized[1]
    service_queue = intitilized[2]
    time_series = intitilized[3]

  
    
    while current_time < stop_time:

        cumulative_propensities = calculate_propensities(state_vector, reactions_list)
        #print("cumulative_propensities")         
        #print(cumulative_propensities)
        next_event_time = draw_next_event_time(current_time, cumulative_propensities)
        if reaction_will_complete(service_queue, next_event_time):
            [state_vector, current_time] = trigger_next_reaction(service_queue, state_vector)
            time_series = write_to_time_series(time_series, current_time, state_vector)
            continue
        current_time = next_event_time
        next_reaction = choose_reaction(cumulative_propensities, reactions_list)
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            time_series = write_to_time_series(time_series, current_time, state_vector)
        else:
            add_reaction(service_queue, current_time + processing_time, next_reaction)
        #print(cumulative_propensities)    
    return dataframe_to_numpyarray(time_series)
    #return 0

def initialize(initial_state_vector):
    state_vector = initial_state_vector
    current_time = 0
    service_queue = []
    time_series = pd.DataFrame([[current_time, state_vector]], columns=['time', 'state'])
    return (state_vector, current_time, service_queue, time_series)



''' calculate_propensities creates an array with the cumulative sum of the propensity functions. '''


def calculate_propensities(x, reactions_list):
    propensities = np.zeros(np.shape(reactions_list))
    for index in range(np.size(reactions_list)):
        propensities[index] = reactions_list[index].propensity(x)
    return np.cumsum(propensities)


def reaction_will_complete(queue, next_event_time):
    if len(queue) > 0:
        if next_event_time > queue[0].comp_time:
            return True
    return False


def draw_next_event_time(current_time, cumulative_propensities):
    #print(cumulative_propensities[0])# debug
    return current_time + np.random.exponential(scale=(1 / cumulative_propensities[-1]))


''' choose_reaction rolls a biased die to determine which reaction will take place or be scheduled next.
    Simple as it gets, as optimal as it gets... I think. '''


def choose_reaction(cumulative_propensities, reactions_list):
    u = np.random.uniform()
    next_reaction_index = min(np.where(cumulative_propensities > cumulative_propensities[-1] * u)[0])
    return reactions_list[next_reaction_index]


''' add_reaction, while not a pure function, does what it is supposed to,
    inserts into the queue a new delayed reaction sorted by completion time. '''


def add_reaction(queue, schedule_time, next_reaction):
    reaction = ScheduleChange(schedule_time, next_reaction.change_vec)
    if len(queue) == 0:
        return queue.append(reaction)
    else:
        for k in range(len(queue)):
            if reaction.comp_time < queue[k].comp_time:
                return queue.insert(k, reaction)
    return queue.append(reaction)


''' trigger_next_reaction has the side effect of removing the first entry of the queue it was passed. '''


def trigger_next_reaction(queue, state_vector):
    next_reaction = queue.pop(0)
    state_vector = state_vector + next_reaction.change_vec
    current_time = next_reaction.comp_time
    return [state_vector, current_time]


def write_to_time_series(time_series, current_time, state_vector):
    return time_series.append(pd.DataFrame([[current_time, state_vector]],
                                           columns=['time', 'state']), ignore_index=True)


''' dataframe_to_numpyarray allows us to use the more efficient DataFrame class to record time series
    and then convert that object back into a usable numpy array. '''


def dataframe_to_numpyarray(framed_data):
    timestamps = np.array(framed_data[['time']])
    states = framed_data[['state']]
    arrayed_data = np.zeros([max(np.shape(timestamps)), np.shape(states.iloc[0, 0])[0] + 1])
    arrayed_data[:, 0] = timestamps.transpose()
    for index in range(max(np.shape(timestamps))):
        arrayed_data[index, 1:] = states.iloc[index, 0]
    return arrayed_data

class Reaction:
    propensities_list = ['mobius_propensity', 'decreasing_hill_propensity',
                         'increasing_hill_propensity', 'mobius_sum_propensity',
                         'dual_feedback_decreasing_hill_propensity', 'dual_feedback_increasing_hill_propensity']
    distributions_list = ['gamma_distribution', 'trivial_distribution', 'bernoulli_distribution']
    system_size = 1

    ''' DO NOT EDIT THIS SECTION '''

    def __init__(self, state_change_vector, parts_of_state_vector,
                 propensity_id, propensity_params,
                 distribution_id, distribution_params):
        self.change_vec = state_change_vector
        self.parts_of_vec = parts_of_state_vector
        if type(propensity_id) == str:
            self.prop_id = Reaction.propensities_list.index(propensity_id)
        else:
            self.prop_id = propensity_id
        self.prop_par = propensity_params
        if type(distribution_id) == str:
            self.dist_id = Reaction.distributions_list.index(distribution_id)
        else:
            self.dist_id = distribution_id
        self.dist_par = distribution_params

    ''' DO NOT EDIT THIS SECTION '''

    def propensity(self, x):
        return getattr(self, Reaction.propensities_list[self.prop_id])(x * Reaction.system_size) / Reaction.system_size

    def distribution(self):
        return getattr(self, Reaction.distributions_list[self.dist_id])()

    ''' Propensities start here.
    After adding a propensity function to the list of definitions, 
    append the name to props_list. '''

    def mobius_propensity(self, state_vector):
        """ For a constant function f(x) = c, assign the vector [c,0,1,0]
            For a linear map f(x) = a * x,    assign the vector [0,a,1,0] """
        x = state_vector[self.parts_of_vec]
        return (self.prop_par[0] + self.prop_par[1] * x) / (self.prop_par[2] + self.prop_par[3] * x)

    def decreasing_hill_propensity(self, state_vector):
        x = state_vector[self.parts_of_vec]
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        exponent = self.prop_par[2]
        return scale * (threshold / (x + threshold)) ** exponent

    def increasing_hill_propensity(self, state_vector):
        x = state_vector[self.parts_of_vec]
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        exponent = self.prop_par[2]
        return scale * (x / (x + threshold)) ** exponent

    def mobius_sum_propensity(self, state_vector):
        total_species = np.sum(state_vector)
        x = state_vector[self.parts_of_vec]
        return (self.prop_par[0] + self.prop_par[1] * x) / (self.prop_par[2] + self.prop_par[3] * total_species)

    def dual_feedback_decreasing_hill_propensity(self, state_vector):
        scale = self.prop_par[0]
        threshold0 = self.prop_par[1]
        threshold1 = self.prop_par[2]
        factor = self.prop_par[3]
        exponent = self.prop_par[4]
        return scale * (threshold0 / (threshold0 + state_vector[self.parts_of_vec])) ** exponent * \
               (threshold1 / factor + state_vector[(self.parts_of_vec + 1) % 2]) / \
               (threshold1 + state_vector[(self.parts_of_vec + 1) % 2])

    def dual_feedback_increasing_hill_propensity(self, state_vector):
        scale = self.prop_par[0]
        threshold0 = self.prop_par[1]
        threshold1 = self.prop_par[2]
        factor = self.prop_par[3]
        exponent = self.prop_par[4]
        return scale * (threshold0 / (threshold0 + state_vector[(self.parts_of_vec + 1) % 2])) ** exponent * \
               (threshold1 / factor + state_vector[self.parts_of_vec]) / \
               (threshold1 + state_vector[self.parts_of_vec])

    def heavyside_propensity(self, state_vector):
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        

    ''' Distributions start here.
    After adding a distribution to the list of definitions,
    append the name to distr_list. '''

    def gamma_distribution(self):
        if self.dist_par[1] == 0:
            return self.trivial_distribution()
        mean = self.dist_par[0]
        stdev = self.dist_par[1]
        return random.gamma(shape=(mean / stdev) ** 2, scale=stdev ** 2 / mean)

    def trivial_distribution(self):
        mean = self.dist_par[0]
        return mean

    def bernoulli_distribution(self):
        mean = self.dist_par[0]
        stdev = self.dist_par[1]
        return mean - 1 + 2 * stdev * random.randint(2)


class ScheduleChange:
    def __init__(self, completion_time, change_vector):
        self.comp_time = completion_time
        self.change_vec = change_vector

mean_range = np.linspace(2, 10, 16)
cv_range = np.linspace(0, .5, 16)
alpha = 300
beta = .1
R0 = 1
C0 = 10
yr =40
yr_range = np.linspace(40, 160, 2)  # gamma
param = 'beta'

for yr in yr_range:

    path1 = 'PostProcessing/Simulations/yr' + str(yr)
    Path(path1).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([mean_range]).to_csv(path1 + '/0metadata.csv', header=False, index=False)
    pd.DataFrame([cv_range]).to_csv(path1 + '/0metadata.csv', mode='a', header=False, index=False)


    row_of_file_names = []
    for mu in mean_range:
        for cv in cv_range:
            file_name = 'PostProcessing/Simulations/yr' + str(yr) + '/mean=' + str(mu) + '_CV=' + str(cv) + '_yr=' + str(yr) + '.csv'
            row_of_file_names.append(file_name)
        paths = path1 + '/1metadata.csv'
        pd.DataFrame([row_of_file_names]).to_csv(paths,  mode='a', header=False, index=False)
        # pd.DataFrame([row_of_file_names]).to_csv('PostProcessing/Simulations/yr/1metadata.csv', mode='a', header=False, index=False)
        row_of_file_names = []

    dilution = Reaction(np.array([-1, 0], dtype=int), 0, 0, [0, beta, 1, 0], 1, [0])


    def gillespie_sim(mu, cv, alpha, beta, R0, C0, yr):
        init_Protein = (alpha - yr) * (
                    mu - C0 * (math.sqrt(alpha / yr) - 1) / yr)  # calculate the avg peak to initialize at a peak
        production = Reaction(np.array([1, 0], dtype=int), 0, 1, [alpha, C0, 2], 0, [mu, mu * cv])
        time_series = gillespie(np.array([production, enzymatic_degradation, dilution]), 5,
                                    np.array([init_Protein, 0], dtype=int))
        file_name =  'PostProcessing/Simulations/yr' + str(yr) + '/mean=' + str(mu) + '_CV=' + str(cv) + '_yr=' + str(yr) + '.csv'
        #badWords = open(file_name, 'w')  # make sure there is an empty file there
        #badWords.close()
        pd.DataFrame(time_series).to_csv(file_name, header=False, index=False)


    safeProcessors = max(1, mp.cpu_count() - 1)
    pool = mp.Pool(safeProcessors)
    # for yr in yr_range:
    for mu in mean_range:

        enzymatic_degradation = Reaction(np.array([-1, 0], dtype=int), 0, 0, [0, yr, R0, 1], 1, [0])

        for cv in cv_range:
            pool.apply_async(gillespie_sim, args=[mu, cv,alpha,beta,R0 ,C0,yr ])
            #gillespie_sim(mu, cv, alpha, beta, R0, C0, yr)
    
    time.sleep(120) 
    pool.close()
    pool.join()
