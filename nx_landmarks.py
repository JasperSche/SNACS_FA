#Class to initialize a landmarks object
#
#
#
import networkx as nx
import numpy as np


class selection_strategies:    
    def __init__(self):
        pass
    def degree_ranking(G:nx.classes.graph.Graph)->list:
        degree = sorted([(v,(G.degree(v))) for v in G], key=lambda tup: tup[1], reverse=True)
        return [d[0] for d in degree]
    def closeness_ranking(G:nx.classes.graph.Graph)->list:
        closeness = nx.closeness_centrality(G)
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
    def __init__(self, G:nx.classes.graph.Graph, d:int = 1, selection_strategie:str = "rand", h:int = 0):
        self.graph = G
        self.d = d
        self.h = h
        supported_rankings = {
            'rand': selection_strategies.random_ranking,
            'deg': selection_strategies.degree_ranking,
            'close': selection_strategies.closeness_ranking,
            'between':selection_strategies.betweeness_ranking
        }

        self.selection_strategie = supported_rankings[selection_strategie]
        self.landmark_ranking = self.selection_strategie(G)
        self.landmarks = None
        self.embeddings = None

    def get_landmarks(self):
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

    def mixed_strategies_init():
        landmarks = []
        return landmarks

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