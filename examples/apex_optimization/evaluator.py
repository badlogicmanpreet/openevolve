"""
Evaluator for Apex optimization example

This evaluator analyzes Apex code patterns and simulates performance metrics
relevant to Salesforce development, including governor limits and best practices.
"""

import importlib.util
import re
import time
import tempfile
import os
import ast
import json
import traceback
from typing import Dict, List, Tuple, Any
import subprocess


class ApexCodeAnalyzer:
    """Analyzes Apex code for performance patterns and complexity"""
    
    def __init__(self):
        # Performance anti-patterns to detect
        self.soql_in_loop_pattern = r'for\s*\([^)]*\)\s*\{[^}]*\[SELECT[^}]*\}'
        self.dml_in_loop_pattern = r'for\s*\([^)]*\)\s*\{[^}]*(?:insert|update|delete|upsert)[^}]*\}'
        self.nested_loop_pattern = r'for\s*\([^)]*\)\s*\{[^}]*for\s*\([^)]*\)[^}]*\}'
        
    def analyze_code_structure(self, apex_code: str) -> Dict[str, Any]:
        """Analyze the structure and patterns in Apex code"""
        
        # Remove comments and strings to avoid false positives
        cleaned_code = self._clean_code(apex_code)
        
        analysis = {
            'lines_of_code': len(apex_code.split('\n')),
            'cyclomatic_complexity': self._calculate_complexity(cleaned_code),
            'soql_queries': self._count_soql_queries(cleaned_code),
            'dml_operations': self._count_dml_operations(cleaned_code),
            'nested_loops': self._count_nested_loops(cleaned_code),
            'soql_in_loops': self._count_soql_in_loops(cleaned_code),
            'dml_in_loops': self._count_dml_in_loops(cleaned_code),
            'method_count': self._count_methods(cleaned_code),
            'loop_count': self._count_loops(cleaned_code),
            'conditional_count': self._count_conditionals(cleaned_code),
        }
        
        return analysis
    
    def _clean_code(self, code: str) -> str:
        """Remove comments and string literals to avoid false positives"""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove string literals
        code = re.sub(r"'[^']*'", "''", code)
        code = re.sub(r'"[^"]*"', '""', code)
        return code
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += len(re.findall(r'\bif\b', code, re.IGNORECASE))
        complexity += len(re.findall(r'\belse\s+if\b', code, re.IGNORECASE))
        complexity += len(re.findall(r'\bwhile\b', code, re.IGNORECASE))
        complexity += len(re.findall(r'\bfor\b', code, re.IGNORECASE))
        complexity += len(re.findall(r'\bcatch\b', code, re.IGNORECASE))
        complexity += len(re.findall(r'\bcase\b', code, re.IGNORECASE))
        complexity += len(re.findall(r'\?\s*[^:]+\s*:', code))  # Ternary operators
        
        return complexity
    
    def _count_soql_queries(self, code: str) -> int:
        """Count SOQL queries in the code"""
        return len(re.findall(r'\[SELECT', code, re.IGNORECASE))
    
    def _count_dml_operations(self, code: str) -> int:
        """Count DML operations"""
        dml_pattern = r'\b(?:insert|update|delete|upsert)\s+'
        return len(re.findall(dml_pattern, code, re.IGNORECASE))
    
    def _count_nested_loops(self, code: str) -> int:
        """Count nested loop structures"""
        return len(re.findall(self.nested_loop_pattern, code, re.IGNORECASE | re.DOTALL))
    
    def _count_soql_in_loops(self, code: str) -> int:
        """Count SOQL queries inside loops (N+1 problem)"""
        return len(re.findall(self.soql_in_loop_pattern, code, re.IGNORECASE | re.DOTALL))
    
    def _count_dml_in_loops(self, code: str) -> int:
        """Count DML operations inside loops"""
        return len(re.findall(self.dml_in_loop_pattern, code, re.IGNORECASE | re.DOTALL))
    
    def _count_methods(self, code: str) -> int:
        """Count methods/functions"""
        return len(re.findall(r'(?:public|private|global|protected)?\s*static?\s*\w+\s+\w+\s*\(', code))
    
    def _count_loops(self, code: str) -> int:
        """Count total loops"""
        for_loops = len(re.findall(r'\bfor\s*\(', code, re.IGNORECASE))
        while_loops = len(re.findall(r'\bwhile\s*\(', code, re.IGNORECASE))
        return for_loops + while_loops
    
    def _count_conditionals(self, code: str) -> int:
        """Count conditional statements"""
        return len(re.findall(r'\bif\s*\(', code, re.IGNORECASE))


class ApexPerformanceSimulator:
    """Simulates Apex performance metrics based on code analysis"""
    
    def __init__(self):
        # Performance cost constants (based on Salesforce governor limits)
        self.soql_cost = 10  # CPU units per SOQL query
        self.dml_cost = 15   # CPU units per DML operation
        self.loop_cost = 2   # CPU units per loop iteration
        self.nested_loop_multiplier = 5  # Additional cost for nested loops
        self.soql_in_loop_penalty = 50  # Heavy penalty for SOQL in loops
        self.dml_in_loop_penalty = 30   # Heavy penalty for DML in loops
        
    def simulate_performance(self, analysis: Dict[str, Any], record_count: int = 50) -> Dict[str, float]:
        """Simulate performance metrics based on code analysis"""
        
        # Base CPU time calculation
        base_cpu_time = 100  # Base execution time in milliseconds
        
        # Calculate CPU time based on operations
        cpu_time = base_cpu_time
        cpu_time += analysis['soql_queries'] * self.soql_cost
        cpu_time += analysis['dml_operations'] * self.dml_cost
        cpu_time += analysis['loop_count'] * self.loop_cost * record_count
        cpu_time += analysis['nested_loops'] * self.nested_loop_multiplier * record_count * record_count
        
        # Heavy penalties for anti-patterns
        cpu_time += analysis['soql_in_loops'] * self.soql_in_loop_penalty * record_count
        cpu_time += analysis['dml_in_loops'] * self.dml_in_loop_penalty * record_count
        
        # Governor limit simulation
        projected_soql_queries = analysis['soql_queries']
        if analysis['soql_in_loops'] > 0:
            projected_soql_queries = analysis['soql_in_loops'] * record_count
            
        projected_dml_operations = analysis['dml_operations']
        if analysis['dml_in_loops'] > 0:
            projected_dml_operations = analysis['dml_in_loops'] * record_count
        
        # Memory usage simulation (heap size)
        memory_usage = 1000  # Base memory in KB
        memory_usage += record_count * 2  # Memory per record
        memory_usage += analysis['soql_queries'] * 50  # Memory for query results
        memory_usage += analysis['nested_loops'] * record_count * 5  # Memory for nested operations
        
        return {
            'cpu_time_ms': cpu_time,
            'memory_usage_kb': memory_usage,
            'projected_soql_queries': projected_soql_queries,
            'projected_dml_operations': projected_dml_operations,
            'execution_efficiency': self._calculate_efficiency(cpu_time, record_count),
            'governor_limit_risk': self._calculate_governor_risk(projected_soql_queries, projected_dml_operations),
        }
    
    def _calculate_efficiency(self, cpu_time: float, record_count: int) -> float:
        """Calculate efficiency score (higher is better)"""
        if cpu_time <= 0:
            return 0.0
        
        # Efficiency = records processed per unit of CPU time
        base_efficiency = (record_count * 1000) / cpu_time
        return min(base_efficiency, 10.0)  # Cap at 10.0
    
    def _calculate_governor_risk(self, soql_queries: int, dml_operations: int) -> float:
        """Calculate risk of hitting governor limits (lower is better)"""
        soql_risk = min(soql_queries / 100.0, 1.0)  # 100 SOQL limit
        dml_risk = min(dml_operations / 150.0, 1.0)  # 150 DML limit
        return (soql_risk + dml_risk) / 2.0


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate an Apex program for performance and best practices
    
    Args:
        program_path: Path to the Apex program file
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        # Read the Apex code
        with open(program_path, 'r') as f:
            apex_code = f.read()
        
        # Initialize analyzers
        analyzer = ApexCodeAnalyzer()
        simulator = ApexPerformanceSimulator()
        
        # Run multiple simulations with different record counts
        record_counts = [10, 50, 100, 200]
        all_results = []
        
        for record_count in record_counts:
            start_time = time.time()
            
            # Analyze code structure
            analysis = analyzer.analyze_code_structure(apex_code)
            
            # Simulate performance
            performance = simulator.simulate_performance(analysis, record_count)
            
            analysis_time = time.time() - start_time
            
            # Calculate scores for this record count
            result = {
                'record_count': record_count,
                'analysis': analysis,
                'performance': performance,
                'analysis_time': analysis_time,
            }
            all_results.append(result)
        
        # Calculate aggregate metrics
        return _calculate_aggregate_metrics(all_results, apex_code)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            'cpu_efficiency_score': 0.0,
            'memory_efficiency_score': 0.0,
            'governor_compliance_score': 0.0,
            'code_quality_score': 0.0,
            'scalability_score': 0.0,
            'overall_score': 0.0,
            'error': str(e),
        }


def _calculate_aggregate_metrics(results: List[Dict], apex_code: str) -> Dict[str, float]:
    """Calculate aggregate performance metrics from multiple test runs"""
    
    # Extract performance data
    cpu_times = [r['performance']['cpu_time_ms'] for r in results]
    memory_usage = [r['performance']['memory_usage_kb'] for r in results]
    efficiencies = [r['performance']['execution_efficiency'] for r in results]
    governor_risks = [r['performance']['governor_limit_risk'] for r in results]
    
    # Get analysis from the largest test case
    largest_test = max(results, key=lambda x: x['record_count'])
    analysis = largest_test['analysis']
    
    # Calculate CPU efficiency score (lower CPU time is better)
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    cpu_efficiency_score = max(0.0, 1.0 - (avg_cpu_time / 10000.0))  # Normalize to 0-1
    
    # Calculate memory efficiency score
    avg_memory = sum(memory_usage) / len(memory_usage)
    memory_efficiency_score = max(0.0, 1.0 - (avg_memory / 6000.0))  # 6MB heap limit
    
    # Calculate governor compliance score (inverse of risk)
    avg_governor_risk = sum(governor_risks) / len(governor_risks)
    governor_compliance_score = 1.0 - avg_governor_risk
    
    # Calculate code quality score based on best practices
    code_quality_score = _calculate_code_quality_score(analysis)
    
    # Calculate scalability score (how well it handles increasing data)
    scalability_score = _calculate_scalability_score(results)
    
    # Calculate overall score with weights
    overall_score = (
        0.25 * cpu_efficiency_score +
        0.20 * memory_efficiency_score +
        0.25 * governor_compliance_score +
        0.20 * code_quality_score +
        0.10 * scalability_score
    )
    
    # Additional detailed metrics
    anti_pattern_penalty = _calculate_anti_pattern_penalty(analysis)
    best_practices_bonus = _calculate_best_practices_bonus(apex_code)
    
    return {
        'cpu_efficiency_score': float(cpu_efficiency_score),
        'memory_efficiency_score': float(memory_efficiency_score),
        'governor_compliance_score': float(governor_compliance_score),
        'code_quality_score': float(code_quality_score),
        'scalability_score': float(scalability_score),
        'overall_score': float(overall_score),
        'anti_pattern_penalty': float(anti_pattern_penalty),
        'best_practices_bonus': float(best_practices_bonus),
        'avg_cpu_time_ms': float(avg_cpu_time),
        'avg_memory_usage_kb': float(avg_memory),
        'complexity_score': float(1.0 / (1.0 + analysis['cyclomatic_complexity'] / 10.0)),
        'efficiency_consistency': float(1.0 - (max(efficiencies) - min(efficiencies))),
    }


def _calculate_code_quality_score(analysis: Dict[str, Any]) -> float:
    """Calculate code quality score based on analysis"""
    score = 1.0
    
    # Penalize anti-patterns
    if analysis['soql_in_loops'] > 0:
        score -= 0.3  # Heavy penalty for SOQL in loops
    if analysis['dml_in_loops'] > 0:
        score -= 0.2  # Penalty for DML in loops
    if analysis['nested_loops'] > 2:
        score -= 0.2  # Penalty for excessive nesting
    if analysis['cyclomatic_complexity'] > 15:
        score -= 0.2  # Penalty for high complexity
    
    return max(0.0, score)


def _calculate_scalability_score(results: List[Dict]) -> float:
    """Calculate how well the algorithm scales with data size"""
    if len(results) < 2:
        return 0.5
    
    # Check if CPU time grows linearly, quadratically, or worse
    record_counts = [r['record_count'] for r in results]
    cpu_times = [r['performance']['cpu_time_ms'] for r in results]
    
    # Simple linear regression to detect growth pattern
    n = len(record_counts)
    sum_x = sum(record_counts)
    sum_y = sum(cpu_times)
    sum_xy = sum(x * y for x, y in zip(record_counts, cpu_times))
    sum_x2 = sum(x * x for x in record_counts)
    
    if n * sum_x2 - sum_x * sum_x == 0:
        return 0.5
        
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    # Score based on growth rate (lower slope is better)
    if slope < 1.0:
        return 1.0  # Excellent scalability
    elif slope < 5.0:
        return 0.8  # Good scalability
    elif slope < 20.0:
        return 0.5  # Moderate scalability
    else:
        return 0.2  # Poor scalability


def _calculate_anti_pattern_penalty(analysis: Dict[str, Any]) -> float:
    """Calculate penalty for anti-patterns"""
    penalty = 0.0
    penalty += analysis['soql_in_loops'] * 0.3
    penalty += analysis['dml_in_loops'] * 0.2
    penalty += analysis['nested_loops'] * 0.1
    return min(penalty, 1.0)


def _calculate_best_practices_bonus(apex_code: str) -> float:
    """Calculate bonus for following best practices"""
    bonus = 0.0
    
    # Check for bulk operations
    if 'List<' in apex_code and ('insert ' in apex_code or 'update ' in apex_code):
        bonus += 0.2
    
    # Check for proper exception handling
    if 'try' in apex_code and 'catch' in apex_code:
        bonus += 0.1
    
    # Check for efficient query patterns
    if 'WHERE' in apex_code and 'IN :' in apex_code:
        bonus += 0.1
    
    return min(bonus, 0.5)


# Cascade evaluation functions for performance optimization
def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """Stage 1: Quick syntax and basic pattern check"""
    try:
        with open(program_path, 'r') as f:
            apex_code = f.read()
        
        analyzer = ApexCodeAnalyzer()
        analysis = analyzer.analyze_code_structure(apex_code)
        
        # Quick checks
        has_major_issues = (
            analysis['soql_in_loops'] > 0 or
            analysis['dml_in_loops'] > 0 or
            analysis['nested_loops'] > 3
        )
        
        basic_quality = 0.8 if not has_major_issues else 0.3
        
        return {
            'basic_quality_check': basic_quality,
            'syntax_valid': 1.0,  # Assume valid if we can parse it
            'has_evolve_blocks': 1.0 if 'EVOLVE-BLOCK-START' in apex_code else 0.0,
        }
        
    except Exception as e:
        return {
            'basic_quality_check': 0.0,
            'syntax_valid': 0.0,
            'error': str(e),
        }


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """Stage 2: Full evaluation"""
    return evaluate(program_path) 