# OpenEvolve: Complete System Guide

An open-source implementation of evolutionary programming for automated algorithm design and code optimization.

## Table of Contents

- [OpenEvolve Architecture: Complete System Explanation](#openevolve-architecture-complete-system-explanation)
- [Core Components Deep Dive](#core-components-deep-dive)
- [Function Minimization Example](#function-minimization-example)
- [Apex Optimization Example](#apex-optimization-example)
- [Evolution Workflow](#evolution-workflow)
- [Advanced Features](#advanced-features)

---

# OpenEvolve Architecture: Complete System Explanation

Based on the architecture diagram and code analysis, here's how OpenEvolve works as an evolutionary coding system:

## Overall Architecture Overview

OpenEvolve implements an **asynchronous evolutionary pipeline** with four core components working in coordination:

1. **Controller** (Orchestration Hub) - Central coordinator
2. **Program Database** - Storage and selection engine
3. **LLM Ensemble** - Code generation engine
4. **Prompt Sampler** - Context creation engine
5. **Evaluator Pool** - Testing and scoring engine

![OpenEvolve Architecture](openevolve-architecture.png)

---

# Core Components Deep Dive

## 1. Controller (Orchestration Hub)

**Purpose**: The central orchestrator that coordinates all components in the evolutionary loop.

**Key Responsibilities**:
```python
class OpenEvolve:
    def __init__(self, initial_program_path, evaluation_file, config_path):
        # Initialize all components
        self.llm_ensemble = LLMEnsemble(config.llm.models)
        self.database = ProgramDatabase(config.database)
        self.evaluator = Evaluator(config.evaluator, evaluation_file)
        self.prompt_sampler = PromptSampler(config.prompt)
```

**Main Evolution Loop**:
```python
async def run(self, iterations=1000):
    for i in range(iterations):
        # 1. SAMPLE: Select parent + inspirations from database
        parent, inspirations = self.database.sample()
        
        # 2. PROMPT: Create context-rich prompt
        prompt = self.prompt_sampler.build_prompt(
            current_program=parent.code,
            program_metrics=parent.metrics,
            top_programs=inspirations
        )
        
        # 3. GENERATE: Use LLM to create code modification
        response = await self.llm_ensemble.generate_with_context(prompt)
        
        # 4. APPLY: Parse and apply changes to create child
        child_code = apply_diff(parent.code, response)
        
        # 5. EVALUATE: Test the child program
        metrics = await self.evaluator.evaluate_program(child_code)
        
        # 6. UPDATE: Add to database and track progress
        child_program = Program(code=child_code, metrics=metrics)
        self.database.add(child_program)
```

## 2. Program Database (Storage & Selection Engine)

**Purpose**: Intelligent storage system that maintains program diversity and enables smart selection.

### MAP-Elites Algorithm
```python
class ProgramDatabase:
    def __init__(self, config):
        self.feature_map: Dict[str, str] = {}  # Grid of feature combinations
        self.feature_bins = 10  # Grid resolution
        self.programs: Dict[str, Program] = {}  # All programs
```

- **Feature Dimensions**: `["score", "complexity"]` - Creates 10x10 grid
- **Diversity Maintenance**: Each grid cell stores the best program for that feature combination
- **Prevents Convergence**: Forces exploration of different algorithmic approaches

### Island Model Evolution
```python
self.islands: List[Set[str]] = [set() for _ in range(5)]  # 5 separate populations
self.current_island: int = 0  # Current active island
```

- **5 Parallel Populations**: Each evolves independently
- **Migration**: Best programs periodically migrate between islands
- **Diversity**: Prevents single evolutionary strategy dominance

### Multi-Tier Selection Strategy
```python
def sample(self) -> Tuple[Program, List[Program]]:
    # 10% elite selection (absolute best)
    # 70% exploitation (good performers)  
    # 20% exploration (diverse solutions)
```

### Best Program Tracking
```python
self.best_program_id: Optional[str] = None  # Tracks absolute best
def _update_best_program(self, program: Program):
    # Ensures best solution is never lost during evolution
```

## 3. LLM Ensemble (Code Generation Engine)

**Purpose**: Manages multiple LLM models to generate diverse code modifications.

**Ensemble Configuration**:
```python
class LLMEnsemble:
    def __init__(self, models_cfg):
        self.models = [OpenAILLM(cfg) for cfg in models_cfg]
        # Example: 80% GPT-4o, 20% GPT-4o-mini
        self.weights = [0.8, 0.2]
```

**Generation Process**:
```python
async def generate_with_context(self, system_message, messages):
    model = self._sample_model()  # Weighted selection
    return await model.generate(system_message, messages)
```

**Benefits**:
- **Diversity**: Different models provide different perspectives
- **Reliability**: Fallback if primary model fails
- **Performance**: Load balancing across models

## 4. Prompt Sampler (Context Creation Engine)

**Purpose**: Creates rich, context-aware prompts that guide effective code evolution.

**Prompt Structure**:
```python
def build_prompt(self, current_program, program_metrics, top_programs):
    prompt = {
        "system": "You are an expert programmer...",
        "user": f"""
        Current Program Metrics: {format_metrics(program_metrics)}
        
        Improvement Areas: {identify_improvement_areas()}
        
        Evolution History:
        {format_previous_attempts()}
        
        Top Performing Programs:
        {format_top_programs(top_programs)}
        
        Current Program:
        ```python
        {current_program}
        ```
        
        Please improve this code...
        """
    }
```

### Advanced Features

#### Template Stochasticity
```python
template_variations = {
    "improvement_suggestion": [
        "Here's how we could improve this code:",
        "I suggest the following improvements:", 
        "We can enhance this code by:"
    ]
}
```

#### Evolution History Context
- **Previous Attempts**: Shows what was tried and results
- **Top Programs**: Provides examples of successful solutions
- **Metrics Analysis**: Identifies patterns in performance

#### Improvement Area Analysis
- **Performance Regression Detection**: Identifies declining metrics
- **Code Complexity Analysis**: Suggests simplification
- **Trend Analysis**: Learns from evolution trajectory

## 5. Evaluator Pool (Testing & Scoring Engine)

**Purpose**: Executes programs safely and assigns comprehensive performance scores.

**Evaluation Architecture**:
```python
class Evaluator:
    def __init__(self, config, evaluation_file):
        self.task_pool = TaskPool(max_concurrency=4)  # Parallel evaluation
        self.evaluate_function = load_function(evaluation_file)
```

### Cascade Evaluation (Performance Optimization)
```python
async def _cascade_evaluate(self, program_path):
    # Stage 1: Quick basic tests (threshold: 0.5)
    stage1_result = await evaluate_stage1(program_path)
    if not passes_threshold(stage1_result, 0.5):
        return stage1_result
    
    # Stage 2: More comprehensive tests (threshold: 0.75)
    stage2_result = await evaluate_stage2(program_path)
    if not passes_threshold(stage2_result, 0.75):
        return merge_results(stage1_result, stage2_result)
    
    # Stage 3: Full evaluation suite
    stage3_result = await evaluate_stage3(program_path)
    return merge_all_results(stage1_result, stage2_result, stage3_result)
```

**Benefits**:
- **Efficiency**: Poor programs fail early, saving computation
- **Thoroughness**: Good programs get comprehensive testing
- **Scalability**: Handles large populations efficiently

### Safety & Robustness
```python
async def evaluate_program(self, program_code, program_id):
    # Timeout protection (5 seconds default)
    # Retry logic (3 attempts)
    # Exception handling
    # Resource isolation
```

### Multi-Metric Scoring
```python
# Example metrics from function minimization:
{
    "value_score": 0.990,      # How close to target value
    "distance_score": 0.921,   # How close to target location  
    "speed_score": 0.466,      # Algorithm efficiency
    "reliability_score": 1.000, # Consistency across runs
    "combined_score": 0.922    # Weighted combination
}
```

## 6. Configuration System

**Purpose**: Unified configuration management for all components.

```python
@dataclass
class Config:
    # Evolution settings
    max_iterations: int = 1000
    diff_based_evolution: bool = True
    allow_full_rewrites: bool = False
    
    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Database configuration  
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Evaluator configuration
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    
    # Prompt configuration
    prompt: PromptConfig = field(default_factory=PromptConfig)
```

---

# Evolution Workflow

## Complete Evolutionary Workflow

Here's how all components work together in the **Asynchronous Pipeline**:

### Iteration Flow:
```
1. Database.sample() → Parent + Inspirations
2. PromptSampler.build_prompt() → Context-rich prompt  
3. LLMEnsemble.generate() → Code modification
4. Controller.apply_diff() → New program version
5. Evaluator.evaluate() → Performance metrics
6. Database.add() → Update population & features
7. Controller.checkpoint() → Save progress (every 10 iterations)
```

### Parallel Processing:
- **Multiple Islands**: 5 populations evolving simultaneously
- **Parallel Evaluation**: 4 programs evaluated concurrently
- **Async Operations**: Non-blocking I/O for LLM calls

### Adaptive Behavior:
- **Migration**: Best solutions spread between islands
- **Template Variations**: Diverse prompt phrasings
- **Cascade Evaluation**: Adaptive testing depth
- **MAP-Elites**: Automatic diversity maintenance

---

# Function Minimization Example

## How OpenEvolve Transforms Simple Algorithms

The function_minimization example demonstrates how OpenEvolve evolves a simple random search into sophisticated optimization algorithms.

### Problem Setup

**Target Function**
```python
f(x, y) = sin(x) * cos(y) + sin(x*y) + (x² + y²)/20
```

**Challenge**: This is a **complex non-convex function** with:
- **Multiple local minima** that trap simple algorithms
- **Global minimum** at approximately (-1.704, 0.678) with value -1.519
- **Search space** bounded to [-5, 5] for both x and y

### Initial Program (Starting Point)

**Initial Algorithm**:
```python
# EVOLVE-BLOCK-START
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """A simple random search algorithm that often gets stuck in local minima."""
    
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    
    for _ in range(iterations):
        # Simple random search - completely random sampling
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        value = evaluate_function(x, y)
        
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
    
    return best_x, best_y, best_value
# EVOLVE-BLOCK-END
```

**Problems with Initial Algorithm**:
- **Pure random sampling** with no intelligence
- **No local search** around good points
- **No escape mechanism** from local minima
- **No adaptive behavior** based on search progress

### Evaluation System

**Multi-Trial Testing**: 10 independent runs for statistical reliability

**Performance Metrics**:
```python
def evaluate(program_path):
    # Run 10 independent trials for statistical reliability
    for trial in range(10):
        x, y, value = run_program_with_timeout(program_path, timeout=5)
        
        # Calculate distance to known global minimum
        distance_to_global = sqrt((x - (-1.704))² + (y - 0.678)²)
        
        # Record results for averaging
        values.append(value)
        distances.append(distance_to_global)
        times.append(execution_time)
    
    # Convert to scores (higher = better)
    value_score = 1.0 / (1.0 + abs(avg_value - (-1.519)))
    distance_score = 1.0 / (1.0 + avg_distance)
    speed_score = min(1.0 / avg_time, 10.0) / 10.0
    reliability_score = success_count / 10
    
    # Weighted combination prioritizing solution quality
    combined_score = (
        0.35 * value_score +      # How close to optimal value
        0.35 * distance_score +   # How close to optimal location  
        0.20 * std_dev_score +    # Consistency across runs
        0.05 * speed_score +      # Algorithm efficiency
        0.05 * reliability_score  # Success rate
    )
```

### Evolution Process in Action

**Typical Evolution Trajectory**:

#### Iteration 1-10: Local Search Discovery
```python
# LLM learns to search around good points instead of pure random
for _ in range(iterations):
    if np.random.random() < 0.7:  # 70% local search
        x = best_x + np.random.normal(0, 0.5)  # Search near best
        y = best_y + np.random.normal(0, 0.5)
    else:  # 30% exploration
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
```

#### Iteration 11-30: Adaptive Behavior
```python
# LLM discovers adaptive step sizes
step_size = initial_step
for i in range(iterations):
    if no_improvement_count > 20:
        step_size *= 1.5  # Expand search if stuck
    if i > iterations * 0.8:
        step_size *= 0.5  # Refine at the end
```

#### Iteration 31-60: Temperature Concepts
```python
# LLM discovers simulated annealing principles
temperature = initial_temperature
for i in range(iterations):
    if new_value > current_value:  # Worse solution
        probability = np.exp((current_value - new_value) / temperature)
        if np.random.random() < probability:
            current_x, current_y = new_x, new_y  # Accept anyway!
    temperature *= cooling_rate  # Cool down over time
```

#### Iteration 61-100: Full Simulated Annealing
```python
# Complete sophisticated algorithm emerges
def search_algorithm(bounds=(-5, 5), iterations=2000, 
                    initial_temperature=100, cooling_rate=0.97,
                    step_size_factor=0.2, step_size_increase_threshold=20):
    # Multi-faceted optimization strategy
    # - Temperature-based acceptance
    # - Adaptive step sizes  
    # - Stagnation handling
    # - Boundary constraints
    # - Cooling schedule
```

### Key Algorithmic Discoveries

OpenEvolve **automatically discovered** these optimization concepts:

#### 1. Temperature-Based Acceptance (Simulated Annealing Core)
```python
# Accepts worse solutions with decreasing probability
probability = np.exp((current_value - new_value) / temperature)
if np.random.rand() < probability:
    current_x, current_y = new_x, new_y  # Escape local minima!
```

#### 2. Adaptive Step Size Management
```python
# Dynamically adjusts exploration vs exploitation
if i > iterations * 0.75:  # Late in search
    step_size *= 0.5  # Focus on refinement
if no_improvement_count > step_size_increase_threshold:  # Stuck
    step_size *= 1.1  # Expand search radius
```

#### 3. Stagnation Detection & Response
```python
# Tracks progress and responds to being stuck
no_improvement_count = 0
if new_value < current_value:
    no_improvement_count = 0  # Reset counter
else:
    no_improvement_count += 1  # Track stagnation
```

#### 4. Boundary Constraint Handling
```python
# Ensures feasible solutions
new_x = max(bounds[0], min(new_x, bounds[1]))
new_y = max(bounds[0], min(new_y, bounds[1]))
```

### Performance Transformation

#### Before Evolution (Random Search):
- **Success Rate**: ~30% (often fails to find good minima)
- **Best Value Found**: ~-0.8 (far from optimal -1.519)
- **Distance to Global**: >3.0 (not in the right region)
- **Reliability**: Poor (high variance between runs)

#### After Evolution (Simulated Annealing):
- **Value Score**: 0.990 (very close to optimal value)
- **Distance Score**: 0.921 (very close to optimal location)
- **Standard Deviation Score**: 0.900 (consistent across runs)
- **Reliability Score**: 1.000 (100% success rate)
- **Overall Score**: 0.984 (excellent performance)

---

# Apex Optimization Example

## Salesforce Apex Data Processing Optimization

This example demonstrates how OpenEvolve can optimize Salesforce Apex code for performance, governor limit compliance, and best practices adherence.

### Problem Description

The task is to optimize an inefficient Apex data processing algorithm that suffers from common Salesforce performance anti-patterns:

- **SOQL queries in loops** (N+1 problem)
- **DML operations in loops** 
- **Excessive nested loops**
- **Poor algorithmic complexity**
- **Governor limit violations**

### Initial Algorithm Issues

The initial `AccountDataProcessor.processAccountData()` method contains multiple performance problems:

```apex
// INEFFICIENT: Individual SOQL query for each account (N+1 problem)
for (Id accountId : accountIds) {
    Account acc = [SELECT Id, Name FROM Account WHERE Id = :accountId];
    
    // INEFFICIENT: Separate query for opportunities for each account
    List<Opportunity> opportunities = [SELECT Id, Amount FROM Opportunity 
                                     WHERE AccountId = :accountId];
    
    // INEFFICIENT: Nested loops with unnecessary complexity
    for (Opportunity opp : opportunities) {
        for (Integer i = 0; i < 10; i++) {
            // Redundant processing
        }
    }
}
```

### Apex Performance Evaluator

The evaluator analyzes Apex code across multiple dimensions:

#### Governor Limit Compliance
- **SOQL Queries**: Must stay under 100 queries per transaction
- **DML Operations**: Must stay under 150 DML statements per transaction
- **CPU Time**: Optimized for minimal processing time
- **Heap Size**: Memory usage efficiency

#### Code Quality Metrics
- **Cyclomatic Complexity**: Measures code complexity
- **Anti-Pattern Detection**: Identifies SOQL/DML in loops
- **Best Practices**: Bulk operations, exception handling
- **Scalability**: Performance with increasing data volumes

#### Performance Scores
```python
{
    'cpu_efficiency_score': float,      # Lower CPU time is better (0.0 - 1.0)
    'memory_efficiency_score': float,   # Efficient memory usage (0.0 - 1.0)
    'governor_compliance_score': float, # Adherence to Salesforce limits (0.0 - 1.0)
    'code_quality_score': float,       # Best practices and patterns (0.0 - 1.0)
    'scalability_score': float,        # Performance scaling with data size (0.0 - 1.0)
    'overall_score': float             # Weighted combination
}
```

### Expected Evolution Patterns

OpenEvolve typically discovers these optimization strategies:

#### Phase 1: Basic Bulk Operations (Iterations 1-30)
```apex
// Evolution learns to bulk query accounts
List<Account> accounts = [SELECT Id, Name FROM Account WHERE Id IN :accountIds];
Map<Id, Account> accountMap = new Map<Id, Account>(accounts);

// Bulk query opportunities
List<Opportunity> allOpportunities = [SELECT AccountId, Amount 
                                     FROM Opportunity 
                                     WHERE AccountId IN :accountIds];
```

#### Phase 2: Data Structure Optimization (Iterations 31-60)
```apex
// Groups opportunities by account for efficient processing
Map<Id, List<Opportunity>> oppsByAccount = new Map<Id, List<Opportunity>>();
for (Opportunity opp : allOpportunities) {
    if (!oppsByAccount.containsKey(opp.AccountId)) {
        oppsByAccount.put(opp.AccountId, new List<Opportunity>());
    }
    oppsByAccount.get(opp.AccountId).add(opp);
}
```

#### Phase 3: Algorithm Optimization (Iterations 61-100)
```apex
// Eliminates nested loops with efficient single-pass processing
for (Account acc : accounts) {
    List<Opportunity> accountOpps = oppsByAccount.get(acc.Id);
    if (accountOpps != null) {
        Decimal totalValue = 0;
        Boolean hasHighValue = false;
        
        // Single pass through opportunities
        for (Opportunity opp : accountOpps) {
            if (opp.Amount != null) {
                totalValue += opp.Amount;
                if (opp.Amount > 100000) {
                    hasHighValue = true;
                }
            }
        }
    }
}
```

#### Phase 4: Advanced Optimization (Iterations 101-150)
```apex
// Discovers advanced patterns like aggregate queries and efficient data structures
public static ProcessingResult processAccountDataOptimized(List<Id> accountIds) {
    // Single comprehensive query with aggregation
    List<AggregateResult> aggResults = [
        SELECT AccountId, SUM(Amount) totalAmount, MAX(Amount) maxAmount
        FROM Opportunity 
        WHERE AccountId IN :accountIds
        AND StageName IN ('Closed Won', 'Negotiation')
        GROUP BY AccountId
    ];
    
    // Process results efficiently without nested loops
    Map<Id, Decimal> totalValueMap = new Map<Id, Decimal>();
    Set<Id> highValueAccounts = new Set<Id>();
    
    for (AggregateResult result : aggResults) {
        Id accountId = (Id)result.get('AccountId');
        Decimal totalAmount = (Decimal)result.get('totalAmount');
        Decimal maxAmount = (Decimal)result.get('maxAmount');
        
        totalValueMap.put(accountId, totalAmount);
        if (maxAmount > 100000 || totalAmount > 500000) {
            highValueAccounts.add(accountId);
        }
    }
    
    // Bulk update accounts
    List<Account> accountsToUpdate = [
        SELECT Id, Premium_Status__c 
        FROM Account 
        WHERE Id IN :highValueAccounts
    ];
    
    for (Account acc : accountsToUpdate) {
        acc.Premium_Status__c = 'High Value';
    }
    
    // Single DML operation
    update accountsToUpdate;
    
    return new ProcessingResult(
        accountsToUpdate.size(),
        aggResults.size() + accountsToUpdate.size(), // Total operations
        2, // SOQL queries
        1  // DML operations
    );
}
```

### Key Optimization Discoveries

Through evolution, OpenEvolve typically discovers:

#### 1. SOQL Optimization
- **Bulk Queries**: Replace individual queries with `WHERE Id IN :collection`
- **Aggregate Queries**: Use `SUM()`, `MAX()`, `COUNT()` for calculations
- **Selective Querying**: Only query needed fields
- **Query Optimization**: Efficient WHERE clauses and indexes

#### 2. DML Best Practices
- **Bulk DML**: Collect records and perform single DML operations
- **Upsert Operations**: Use `upsert` for insert/update scenarios
- **Transaction Management**: Proper savepoint and rollback handling

#### 3. Data Structure Patterns
- **Maps for Lookup**: Use `Map<Id, SObject>` for efficient data access
- **Sets for Uniqueness**: Use `Set<Id>` for unique collections
- **Lists for Processing**: Efficient list operations and iterations

#### 4. Algorithm Efficiency
- **Single-Pass Processing**: Eliminate redundant iterations
- **Early Termination**: Break loops when conditions are met
- **Efficient Sorting**: Use built-in sorting where possible

#### 5. Governor Limit Awareness
- **Query Limits**: Stay under 100 SOQL queries per transaction
- **DML Limits**: Stay under 150 DML operations per transaction
- **CPU Time**: Optimize for quick execution
- **Heap Size**: Efficient memory management

### Performance Improvements

Typical evolution results show dramatic improvements:

| Metric | Initial Code | Evolved Code | Improvement |
|--------|--------------|--------------|-------------|
| CPU Efficiency Score | 0.15 | 0.92 | +513% |
| Memory Efficiency Score | 0.31 | 0.89 | +187% |
| Governor Compliance Score | 0.20 | 0.98 | +390% |
| Code Quality Score | 0.25 | 0.85 | +240% |
| Scalability Score | 0.18 | 0.94 | +422% |
| **Overall Score** | **0.22** | **0.92** | **+318%** |

#### Governor Limit Compliance
- **SOQL Queries**: Reduced from O(n) to O(1) - typically 2 queries total
- **DML Operations**: Reduced from O(n) to O(1) - single bulk operation
- **CPU Time**: Reduced from exponential to linear complexity
- **Memory Usage**: Efficient data structures prevent memory bloat

### Apex-Specific Features

This example includes Salesforce-specific optimizations:

#### Governor Limit Simulation
The evaluator simulates real Salesforce governor limits:
- Projects SOQL/DML usage based on data volume
- Calculates governor limit risk scores
- Penalizes anti-patterns heavily

#### Best Practice Detection
Recognizes Salesforce best practices:
- Bulk operation patterns
- Efficient query structures
- Proper exception handling
- Modular code design

#### Realistic Performance Modeling
Performance simulation based on:
- Actual Salesforce operation costs
- Realistic data volumes (10-200 records)
- Memory usage patterns
- CPU time estimation

---

# Advanced Features

## Key Innovations

1. **Never Lose Best**: Separate tracking ensures optimal solution preservation
2. **Context-Rich Prompts**: Comprehensive evolution history in prompts
3. **Multi-Model Ensemble**: Diverse perspectives from different LLMs
4. **Intelligent Evaluation**: Cascade system optimizes compute usage
5. **Checkpoint Resume**: Complete state preservation for long runs
6. **Island Evolution**: Parallel diversity maintenance

## Getting Started

### Installation
```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start
```python
from openevolve import OpenEvolve

# Initialize the system
evolve = OpenEvolve(
    initial_program_path="path/to/initial_program.py",
    evaluation_file="path/to/evaluator.py",
    config_path="path/to/config.yaml"
)

# Run the evolution
best_program = await evolve.run(iterations=1000)
print(f"Best program metrics:")
for name, value in best_program.metrics.items():
    print(f"  {name}: {value:.4f}")
```

### Running Examples

#### Function Minimization
```bash
cd examples/function_minimization
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

#### Apex Optimization
```bash
cd examples/apex_optimization
python ../../openevolve-run.py initial_program.apex evaluator.py --config config.yaml
```

This architecture enables OpenEvolve to **automatically discover sophisticated algorithms** (like simulated annealing for optimization or bulk operations for Salesforce) starting from simple implementations, making it a powerful tool for automated algorithm design and optimization across multiple programming languages and domains. 