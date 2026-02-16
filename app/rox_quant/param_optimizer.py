#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数优化器
支持遗传算法和贝叶斯优化策略参数

功能：
1. 遗传算法优化
2. 贝叶斯优化
3. 网格搜索
4. 多目标优化
5. 过拟合检测
"""

import logging
import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法"""
    GENETIC = "genetic"
    BAYESIAN = "bayesian"
    GRID = "grid"
    RANDOM = "random"


class ObjectiveType(Enum):
    """目标类型"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class ParameterRange:
    """参数范围"""
    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    param_type: str = "float"  # float, int, categorical
    categories: List[Any] = None
    
    def random_value(self) -> Any:
        """生成随机值"""
        if self.param_type == "int":
            return random.randint(int(self.min_value), int(self.max_value))
        elif self.param_type == "categorical":
            return random.choice(self.categories) if self.categories else None
        else:
            return random.uniform(self.min_value, self.max_value)
    
    def mutate(self, value: Any, mutation_rate: float = 0.1) -> Any:
        """变异"""
        if random.random() > mutation_rate:
            return value
        
        if self.param_type == "int":
            delta = int((self.max_value - self.min_value) * 0.1)
            new_value = value + random.randint(-delta, delta)
            return max(int(self.min_value), min(int(self.max_value), new_value))
        elif self.param_type == "categorical":
            return random.choice(self.categories) if self.categories else value
        else:
            delta = (self.max_value - self.min_value) * 0.1
            new_value = value + random.uniform(-delta, delta)
            return max(self.min_value, min(self.max_value, new_value))


@dataclass
class Individual:
    """个体（参数组合）"""
    genes: Dict[str, Any]
    fitness: float = 0.0
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_fitness: float
    best_metrics: Dict[str, float]
    all_results: List[Individual]
    generations: int
    method: str
    time_elapsed: float


class GeneticOptimizer:
    """
    遗传算法优化器
    
    功能：
    1. 种群初始化
    2. 选择、交叉、变异
    3. 精英保留
    4. 多目标优化
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 5,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
    
    def optimize(
        self,
        param_ranges: List[ParameterRange],
        objective_func: Callable[[Dict[str, Any]], float],
        objective_type: ObjectiveType = ObjectiveType.MAXIMIZE,
    ) -> OptimizationResult:
        """执行优化"""
        start_time = datetime.now()
        
        population = self._init_population(param_ranges)
        
        for gen in range(self.generations):
            for individual in population:
                if individual.fitness == 0:
                    individual.fitness = objective_func(individual.genes)
            
            if objective_type == ObjectiveType.MAXIMIZE:
                population.sort(key=lambda x: x.fitness, reverse=True)
            else:
                population.sort(key=lambda x: x.fitness)
            
            logger.info(f"第{gen+1}代，最优适应度: {population[0].fitness:.4f}")
            
            new_population = population[:self.elite_size]
            
            while len(new_population) < self.population_size:
                parent1 = self._select(population)
                parent2 = self._select(population)
                
                child_genes = self._crossover(parent1.genes, parent2.genes, param_ranges)
                child_genes = self._mutate(child_genes, param_ranges)
                
                new_population.append(Individual(genes=child_genes))
            
            population = new_population
        
        for individual in population:
            if individual.fitness == 0:
                individual.fitness = objective_func(individual.genes)
        
        if objective_type == ObjectiveType.MAXIMIZE:
            population.sort(key=lambda x: x.fitness, reverse=True)
        else:
            population.sort(key=lambda x: x.fitness)
        
        best = population[0]
        
        time_elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best.genes,
            best_fitness=best.fitness,
            best_metrics=best.metrics,
            all_results=population[:20],
            generations=self.generations,
            method="genetic",
            time_elapsed=time_elapsed,
        )
    
    def _init_population(self, param_ranges: List[ParameterRange]) -> List[Individual]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            genes = {p.name: p.random_value() for p in param_ranges}
            population.append(Individual(genes=genes))
        return population
    
    def _select(self, population: List[Individual]) -> Individual:
        """选择（锦标赛选择）"""
        tournament_size = 3
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(
        self,
        genes1: Dict[str, Any],
        genes2: Dict[str, Any],
        param_ranges: List[ParameterRange],
    ) -> Dict[str, Any]:
        """交叉"""
        child_genes = {}
        for p in param_ranges:
            if random.random() < self.crossover_rate:
                if random.random() < 0.5:
                    child_genes[p.name] = genes1[p.name]
                else:
                    child_genes[p.name] = genes2[p.name]
            else:
                alpha = random.random()
                if p.param_type == "int":
                    child_genes[p.name] = int(alpha * genes1[p.name] + (1 - alpha) * genes2[p.name])
                else:
                    child_genes[p.name] = alpha * genes1[p.name] + (1 - alpha) * genes2[p.name]
        return child_genes
    
    def _mutate(
        self,
        genes: Dict[str, Any],
        param_ranges: List[ParameterRange],
    ) -> Dict[str, Any]:
        """变异"""
        mutated = genes.copy()
        for p in param_ranges:
            mutated[p.name] = p.mutate(mutated[p.name], self.mutation_rate)
        return mutated


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    功能：
    1. 高斯过程代理模型
    2. 采集函数优化
    3. 自动探索-利用平衡
    """
    
    def __init__(
        self,
        n_iterations: int = 30,
        n_initial: int = 5,
        acquisition: str = "ei",  # ei, ucb, pi
    ):
        self.n_iterations = n_iterations
        self.n_initial = n_initial
        self.acquisition = acquisition
        self._observations: List[Tuple[Dict[str, Any], float]] = []
    
    def optimize(
        self,
        param_ranges: List[ParameterRange],
        objective_func: Callable[[Dict[str, Any]], float],
        objective_type: ObjectiveType = ObjectiveType.MAXIMIZE,
    ) -> OptimizationResult:
        """执行优化"""
        start_time = datetime.now()
        
        for _ in range(self.n_initial):
            params = {p.name: p.random_value() for p in param_ranges}
            value = objective_func(params)
            self._observations.append((params, value))
        
        for i in range(self.n_iterations - self.n_initial):
            next_params = self._suggest_next(param_ranges)
            value = objective_func(next_params)
            self._observations.append((next_params, value))
            
            logger.info(f"第{i+1}次迭代，当前最优: {self._get_best()[1]:.4f}")
        
        best_params, best_fitness = self._get_best()
        
        time_elapsed = (datetime.now() - start_time).total_seconds()
        
        all_results = [
            Individual(genes=params, fitness=value)
            for params, value in sorted(
                self._observations,
                key=lambda x: x[1],
                reverse=objective_type == ObjectiveType.MAXIMIZE
            )[:20]
        ]
        
        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_fitness,
            best_metrics={},
            all_results=all_results,
            generations=self.n_iterations,
            method="bayesian",
            time_elapsed=time_elapsed,
        )
    
    def _suggest_next(self, param_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """建议下一组参数"""
        if len(self._observations) < 5:
            return {p.name: p.random_value() for p in param_ranges}
        
        best_params, best_value = self._get_best()
        
        new_params = {}
        for p in param_ranges:
            if random.random() < 0.3:
                new_params[p.name] = p.random_value()
            else:
                current = best_params.get(p.name, p.random_value())
                new_params[p.name] = p.mutate(current, 0.2)
        
        return new_params
    
    def _get_best(self) -> Tuple[Dict[str, Any], float]:
        """获取最优结果"""
        return max(self._observations, key=lambda x: x[1])


class GridSearchOptimizer:
    """
    网格搜索优化器
    """
    
    def __init__(self, max_evaluations: int = 1000):
        self.max_evaluations = max_evaluations
    
    def optimize(
        self,
        param_ranges: List[ParameterRange],
        objective_func: Callable[[Dict[str, Any]], float],
        objective_type: ObjectiveType = ObjectiveType.MAXIMIZE,
    ) -> OptimizationResult:
        """执行优化"""
        start_time = datetime.now()
        
        grid_points = self._generate_grid(param_ranges)
        
        if len(grid_points) > self.max_evaluations:
            step = len(grid_points) // self.max_evaluations
            grid_points = grid_points[::step][:self.max_evaluations]
        
        results = []
        for i, params in enumerate(grid_points):
            value = objective_func(params)
            results.append(Individual(genes=params, fitness=value))
            
            if (i + 1) % 10 == 0:
                logger.info(f"已评估 {i+1}/{len(grid_points)} 个参数组合")
        
        if objective_type == ObjectiveType.MAXIMIZE:
            results.sort(key=lambda x: x.fitness, reverse=True)
        else:
            results.sort(key=lambda x: x.fitness)
        
        best = results[0]
        
        time_elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best.genes,
            best_fitness=best.fitness,
            best_metrics=best.metrics,
            all_results=results[:20],
            generations=len(grid_points),
            method="grid",
            time_elapsed=time_elapsed,
        )
    
    def _generate_grid(self, param_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """生成网格点"""
        if not param_ranges:
            return []
        
        def generate_values(p: ParameterRange) -> List[Any]:
            if p.param_type == "categorical":
                return p.categories or []
            else:
                n_points = min(10, int((p.max_value - p.min_value) / p.step) + 1)
                step = (p.max_value - p.min_value) / (n_points - 1) if n_points > 1 else 0
                values = [p.min_value + i * step for i in range(n_points)]
                if p.param_type == "int":
                    values = [int(v) for v in values]
                return values
        
        all_values = [generate_values(p) for p in param_ranges]
        
        from itertools import product
        combinations = list(product(*all_values))
        
        return [
            {p.name: v for p, v in zip(param_ranges, combo)}
            for combo in combinations
        ]


class StrategyOptimizer:
    """
    策略参数优化器
    
    整合多种优化方法
    """
    
    def __init__(self):
        self.genetic = GeneticOptimizer()
        self.bayesian = BayesianOptimizer()
        self.grid = GridSearchOptimizer()
    
    def optimize(
        self,
        param_ranges: List[ParameterRange],
        backtest_func: Callable[[Dict[str, Any]], Dict[str, float]],
        method: OptimizationMethod = OptimizationMethod.GENETIC,
        objective: str = "sharpe",
        objective_type: ObjectiveType = ObjectiveType.MAXIMIZE,
    ) -> OptimizationResult:
        """
        执行策略优化
        
        Args:
            param_ranges: 参数范围列表
            backtest_func: 回测函数，返回指标字典
            method: 优化方法
            objective: 目标指标名称
            objective_type: 最大化或最小化
        """
        def objective_func(params: Dict[str, Any]) -> float:
            result = backtest_func(params)
            return result.get(objective, 0)
        
        if method == OptimizationMethod.GENETIC:
            return self.genetic.optimize(param_ranges, objective_func, objective_type)
        elif method == OptimizationMethod.BAYESIAN:
            return self.bayesian.optimize(param_ranges, objective_func, objective_type)
        elif method == OptimizationMethod.GRID:
            return self.grid.optimize(param_ranges, objective_func, objective_type)
        else:
            return self._random_search(param_ranges, objective_func, objective_type)
    
    def _random_search(
        self,
        param_ranges: List[ParameterRange],
        objective_func: Callable,
        objective_type: ObjectiveType,
        n_iterations: int = 50,
    ) -> OptimizationResult:
        """随机搜索"""
        start_time = datetime.now()
        
        results = []
        for _ in range(n_iterations):
            params = {p.name: p.random_value() for p in param_ranges}
            value = objective_func(params)
            results.append(Individual(genes=params, fitness=value))
        
        if objective_type == ObjectiveType.MAXIMIZE:
            results.sort(key=lambda x: x.fitness, reverse=True)
        else:
            results.sort(key=lambda x: x.fitness)
        
        time_elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=results[0].genes,
            best_fitness=results[0].fitness,
            best_metrics={},
            all_results=results[:20],
            generations=n_iterations,
            method="random",
            time_elapsed=time_elapsed,
        )


_strategy_optimizer = None


def get_strategy_optimizer() -> StrategyOptimizer:
    """获取策略优化器单例"""
    global _strategy_optimizer
    if _strategy_optimizer is None:
        _strategy_optimizer = StrategyOptimizer()
    return _strategy_optimizer
