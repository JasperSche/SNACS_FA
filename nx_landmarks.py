#Class to initialize a landmarks object
#
#
#
import networkx as nx
import numpy as np
import math
import random

class selection_strategies:    
    def __init__(self):
        pass
    
    def degree_ranking(G:nx.classes.graph.Graph)->list:
        degree = sorted([(v,(G.degree(v))) for v in G], key=lambda tup: tup[1], reverse=True)
        return [d[0] for d in degree]
    
    def closeness_ranking(G:nx.classes.graph.Graph, k:int = 500)->list:
        nodes = list(G.nodes())
        sample = random.sample(nodes, min(k, len(nodes)))
        closeness = {v: 0 for v in nodes}
        for s in sample:
            lengths = nx.single_source_shortest_path_length(G, s)
            for v, d in lengths.items():
                closeness[v] += d
                
        closeness = {v: (k / closeness[v]) if closeness[v] > 0 else 0 for v in nodes}
        closeness = sorted([(k,v) for k,v in zip(closeness.keys(), closeness.values())] , key= lambda tup: tup[1], reverse=True)
        return [c[0] for c in closeness]
    
    def random_ranking(G:nx.classes.graph.Graph)->list:
        sample = np.array(G.nodes)
        np.random.shuffle(sample)
        return sample
    
    def betweeness_ranking(G:nx.classes.graph.Graph, k:int=500)->list:
        betweeness = nx.betweenness_centrality(G, 500)
        betweeness = sorted([(k,v) for k,v in zip(betweeness.keys(), betweeness.values())] , key= lambda tup: tup[1], reverse=True)
        return [c[0] for c in betweeness]

class landmarks:
    def __init__(self, G:nx.classes.graph.Graph, d:int = 1, selection_strategie:list = "deg", h:int = 0):
        self.graph = G
        self.d = d
        self.h = h
        self.supported_rankings = {
            'rand': selection_strategies.random_ranking,
            'deg': selection_strategies.degree_ranking,
            'close': selection_strategies.closeness_ranking,
            'between':selection_strategies.betweeness_ranking
        }

        if len(selection_strategie) == 1:
            self.selection_strategie = self.supported_rankings[selection_strategie[0]]
            self.landmark_ranking = self.selection_strategie(G)
        else:
            self.selection_strategie = self.mixed_strategies_init
            self.landmark_ranking = self.selection_strategie(selection_strategie)

        self.landmarks = None
        self.embeddings = None

    def get_landmarks(self):
        if self.landmarks != None:
            self.landmarks = None
            self.embeddings = None 
        landmarks = []
        embeddings = np.full((self.graph.number_of_nodes(),self.d), np.inf)
        x = 0 
        for i in range(self.d):
            while min(embeddings[self.landmark_ranking[x]]) < self.h:
                x+=1
            landmarks.append(self.landmark_ranking[x])   
            x+=1 
            shortest_paths = nx.single_source_shortest_path_length(self.graph, source = landmarks[-1])
            for node, length in shortest_paths.items():
                embeddings[node,i] = length
            
        self.landmarks = np.array(landmarks)
        self.embeddings = embeddings
    
    def shortest_path_estimation_upper_bound(self, source , target):
        uppers = self.embeddings[source] + self.embeddings[target]
        return int(min(uppers))
        
    def shortest_path_estimation_lower_bound(self, source , target):
        lowers = self.embeddings[source] - self.embeddings[target]
        return int(max(abs(lowers)))
    
    def shortest_path_estimation_capture_method(self,source, target):
        lower =  max(abs(self.embeddings[source]-self.embeddings[target]))
        upper =  min(self.embeddings[source]+self.embeddings[target])
        if upper == lower: return upper
        #Find capture upper bound:
        d_bound_lower = np.where(abs(self.embeddings[source]-self.embeddings[source]) == lower) 
        d_bound_upper = np.where((self.embeddings[source]+self.embeddings[source]) == upper)
        for bound in [d_bound_lower,d_bound_upper]:
            #get the 
            pass
            #check weather the captures are found and then confirm / deny lower upper bound 

    def mixed_strategies_init(self,strategies):
        rankings = []
        for strat in strategies:
            this_strat = self.supported_rankings[strat[0]]
            this_ranking = this_strat(self.graph)
            rankings.append((this_ranking, strat[1]))
                    
        landmarks_ranking = {}
        for ranking in rankings:
            ranks  = np.arange(len(ranking[0]))[::-1]
            norm_rank  = (ranks - ranks.min()) / (ranks.max() - ranks.min())
            weighted_rank = norm_rank*ranking[1]
            for idx, node in enumerate(ranking[0]):
                score = weighted_rank[idx]
                if node in landmarks_ranking.keys():
                    if landmarks_ranking[node]< score: landmarks_ranking[node] = score 
                else:
                    landmarks_ranking[node] = score

        landmarks_ranking = [k for k, v in sorted(landmarks_ranking.items(), key=lambda item: item[1], reverse=True)]

        return landmarks_ranking

    def add_landmarks(self, n:int = 1):
        x = np.where(self.landmark_ranking == self.landmarks[-1])
        x = int(x[0])+1
        self.embeddings = np.append(self.embeddings, np.full((self.graph.number_of_nodes(),n), np.inf), axis=1)
        for i in range(self.embeddings.shape[1]-n, self.embeddings.shape[1]):
            while min(self.embeddings[self.landmark_ranking[x]]) < self.h:
                x+=1
            self.landmarks = np.append(self.landmarks, self.landmark_ranking[x])
            x+=1
            shortest_paths = nx.single_source_shortest_path_length(self.graph, source = self.landmarks[-1])
            for node, length in shortest_paths.items():
                self.embeddings[node,i] = length