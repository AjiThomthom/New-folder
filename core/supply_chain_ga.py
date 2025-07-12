import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
import logging

# Konfigurasi logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplyChainOptimizer:
    def __init__(self, n_factories=3, n_markets=5, production_cost=10.0, 
                 transport_rate=0.5, co2_per_km=0.3, max_capacity=1000,
                 min_demand=100, max_demand=500, co2_budget=2000):
        
        self.n_factories = n_factories
        self.n_markets = n_markets
        self.production_cost = production_cost
        self.transport_rate = transport_rate
        self.co2_per_km = co2_per_km
        self.max_capacity = max_capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.co2_budget = co2_budget
        
        # Matriks jarak (akan di-generate secara acak)
        self.distance_matrix = None
        self.cost_matrix = None
        self.emission_matrix = None
        self.demands = None
        self.capacities = None
        
    def generate_network(self):
        # Generate lokasi acak (lat, lon)
        factory_locs = np.random.uniform(
            low=[-6.3, 106.5], 
            high=[-6.0, 107.0], 
            size=(self.n_factories, 2)
        )
        
        market_locs = np.random.uniform(
            low=[-6.4, 106.4], 
            high=[-5.9, 107.2], 
            size=(self.n_markets, 2)
        )
        
        # Hitung matriks jarak
        self.distance_matrix = np.zeros((self.n_factories, self.n_markets))
        for i in range(self.n_factories):
            for j in range(self.n_markets):
                # Hitung jarak Euclidean sederhana
                lat_diff = factory_locs[i,0] - market_locs[j,0]
                lon_diff = factory_locs[i,1] - market_locs[j,1]
                self.distance_matrix[i,j] = np.sqrt(lat_diff**2 + lon_diff**2) * 100  # Skala km
        
        # Matriks biaya dan emisi
        self.cost_matrix = self.distance_matrix * self.transport_rate
        self.emission_matrix = self.distance_matrix * self.co2_per_km
        
        # Generate kapasitas dan permintaan
        self.capacities = np.random.randint(
            self.max_capacity * 0.7, 
            self.max_capacity, 
            self.n_factories
        )
        
        self.demands = np.random.randint(
            self.min_demand, 
            self.max_demand, 
            self.n_markets
        )
        
        # Format data untuk visualisasi
        nodes = []
        for i in range(self.n_factories):
            nodes.append({
                'name': f"Pabrik {i+1}",
                'type': 'factory',
                'capacity': self.capacities[i],
                'lat': factory_locs[i,0],
                'lon': factory_locs[i,1]
            })
            
        for j in range(self.n_markets):
            nodes.append({
                'name': f"Pasar {j+1}",
                'type': 'market',
                'demand': self.demands[j],
                'lat': market_locs[j,0],
                'lon': market_locs[j,1]
            })
            
        edges = []
        for i in range(self.n_factories):
            for j in range(self.n_markets):
                edges.append({
                    'source': f"Pabrik {i+1}",
                    'destination': f"Pasar {j+1}",
                    'distance': self.distance_matrix[i,j],
                    'cost_per_ton': self.cost_matrix[i,j],
                    'co2_per_ton': self.emission_matrix[i,j],
                    'src_lat': factory_locs[i,0],
                    'src_lon': factory_locs[i,1],
                    'dest_lat': market_locs[j,0],
                    'dest_lon': market_locs[j,1]
                })
                
        return {
            'nodes': pd.DataFrame(nodes),
            'edges': pd.DataFrame(edges)
        }
        
    def solve(self, population_size=100, generations=200, 
              mutation_rate=0.1, weights=(0.7, 0.3)):
        
        class SupplyChainProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=self.n_factories * self.n_markets,
                    n_obj=2,
                    n_constr=2,
                    xl=0.0,
                    xu=1.0
                )
                self.weights = weights
                
            def _evaluate(self, X, out, *args, **kwargs):
                n_individuals = X.shape[0]
                F = np.zeros((n_individuals, 2))
                G = np.zeros((n_individuals, 2))
                
                for k in range(n_individuals):
                    # Dekode solusi
                    allocation = X[k].reshape((self.n_factories, self.n_markets))
                    
                    # Hitung produksi total per pabrik
                    production = allocation.sum(axis=1)
                    
                    # Hitung pengiriman total per pasar
                    delivery = allocation.sum(axis=0)
                    
                    # Biaya produksi
                    prod_cost = np.sum(production) * self.production_cost
                    
                    # Biaya transport
                    transport_cost = np.sum(allocation * self.cost_matrix)
                    
                    # Total emisi
                    emissions = np.sum(allocation * self.emission_matrix)
                    
                    # Fungsi objektif
                    F[k, 0] = prod_cost + transport_cost  # Total biaya
                    F[k, 1] = emissions                  # Total emisi
                    
                    # Kendala
                    G[k, 0] = np.maximum(production - self.capacities, 0).sum()  # Kapasitas
                    G[k, 1] = np.maximum(self.demands - delivery, 0).sum()       # Permintaan
                    
                    # Kendala tambahan: batas emisi
                    # G[k, 2] = np.maximum(emissions - self.co2_budget, 0)
                
                # Gabungkan objective dengan bobot
                weighted_F = F[:,0]*self.weights[0] + F[:,1]*self.weights[1]
                out["F"] = weighted_F
                out["G"] = G
        
        # Setup problem
        problem = SupplyChainProblem()
        
        # Konfigurasi algoritma genetika
        algorithm = NSGA2(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=mutation_rate, eta=20),
            eliminate_duplicates=True
        )
        
        # Jalankan optimasi
        res = minimize(problem,
                       algorithm,
                       ('n_gen', generations),
                       seed=1,
                       verbose=False)
        
        # Ambil solusi terbaik
        best_idx = np.argmin(res.F)
        best_solution = res.X[best_idx].reshape((self.n_factories, self.n_markets))
        
        # Skala solusi ke ton
        scaled_solution = best_solution * (self.demands.sum() / best_solution.sum())
        
        # Hitung metrik performa
        total_cost = (scaled_solution * self.cost_matrix).sum() + scaled_solution.sum() * self.production_cost
        total_emission = (scaled_solution * self.emission_matrix).sum()
        utilization = scaled_solution.sum(axis=1) / self.capacities
        
        # Format hasil
        return {
            'allocation': pd.DataFrame(
                scaled_solution,
                index=[f"Pabrik {i+1}" for i in range(self.n_factories)],
                columns=[f"Pasar {j+1}" for j in range(self.n_markets)]
            ),
            'total_cost': total_cost,
            'total_emission': total_emission,
            'utilization': utilization.mean(),
            'convergence': pd.DataFrame({
                'generation': range(len(res.history)),
                'best_cost': [np.min(h.opt.get('F')) for h in res.history],
                'avg_cost': [np.mean(h.opt.get('F')) for h in res.history]
            })
        }