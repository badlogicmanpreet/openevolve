#!/usr/bin/env python3
"""
Test script for the Apex optimization evaluator

This script tests the evaluator with the initial Apex program to ensure
it correctly identifies performance issues and calculates metrics.
"""

import os
import sys
import json
from pathlib import Path

# Add the evaluator to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import evaluate, evaluate_stage1, evaluate_stage2


def test_initial_program():
    """Test evaluation of the initial Apex program"""
    
    print("🧪 Testing Apex optimization evaluator...")
    print("=" * 50)
    
    # Get path to initial program
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.apex")
    
    if not os.path.exists(initial_program_path):
        print(f"❌ Error: Initial program not found at {initial_program_path}")
        return False
    
    print(f"📁 Evaluating: {initial_program_path}")
    
    try:
        # Test Stage 1 evaluation
        print("\n🔍 Stage 1 Evaluation (Quick Check):")
        stage1_results = evaluate_stage1(initial_program_path)
        
        for metric, value in stage1_results.items():
            print(f"  {metric}: {value}")
        
        # Test full evaluation
        print("\n🔍 Full Evaluation:")
        results = evaluate(initial_program_path)
        
        # Display key metrics
        key_metrics = [
            'cpu_efficiency_score',
            'memory_efficiency_score', 
            'governor_compliance_score',
            'code_quality_score',
            'scalability_score',
            'overall_score'
        ]
        
        print("\n📊 Performance Metrics:")
        for metric in key_metrics:
            if metric in results:
                value = results[metric]
                emoji = "✅" if value > 0.7 else "⚠️" if value > 0.4 else "❌"
                print(f"  {emoji} {metric}: {value:.3f}")
        
        # Display additional metrics
        print("\n📈 Additional Metrics:")
        additional_metrics = [
            'avg_cpu_time_ms',
            'avg_memory_usage_kb',
            'anti_pattern_penalty',
            'best_practices_bonus',
            'complexity_score'
        ]
        
        for metric in additional_metrics:
            if metric in results:
                value = results[metric]
                print(f"  • {metric}: {value:.3f}")
        
        # Validate expected issues
        print("\n🔍 Anti-Pattern Detection:")
        
        expected_issues = {
            'low_governor_compliance': results.get('governor_compliance_score', 1.0) < 0.5,
            'poor_code_quality': results.get('code_quality_score', 1.0) < 0.5,
            'high_anti_pattern_penalty': results.get('anti_pattern_penalty', 0.0) > 0.3,
            'poor_scalability': results.get('scalability_score', 1.0) < 0.5
        }
        
        for issue, detected in expected_issues.items():
            emoji = "✅" if detected else "⚠️"
            print(f"  {emoji} {issue}: {'Detected' if detected else 'Not detected'}")
        
        # Overall assessment
        overall_score = results.get('overall_score', 0.0)
        print(f"\n🎯 Overall Assessment:")
        
        if overall_score < 0.3:
            print(f"  ❌ Poor performance (score: {overall_score:.3f}) - Needs significant optimization")
        elif overall_score < 0.6:
            print(f"  ⚠️ Moderate performance (score: {overall_score:.3f}) - Room for improvement")
        else:
            print(f"  ✅ Good performance (score: {overall_score:.3f}) - Well optimized")
        
        # Test summary
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📄 Full results contain {len(results)} metrics")
        
        # Save results for reference
        results_file = "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator_components():
    """Test individual evaluator components"""
    
    print("\n🔧 Testing Evaluator Components:")
    print("=" * 40)
    
    try:
        from evaluator import ApexCodeAnalyzer, ApexPerformanceSimulator
        
        # Test code analyzer
        print("🔍 Testing ApexCodeAnalyzer...")
        
        sample_apex = """
        public class TestClass {
            public static void processData(List<Id> ids) {
                for (Id id : ids) {
                    Account acc = [SELECT Id FROM Account WHERE Id = :id];
                    for (Integer i = 0; i < 10; i++) {
                        if (acc != null) {
                            update acc;
                        }
                    }
                }
            }
        }
        """
        
        analyzer = ApexCodeAnalyzer()
        analysis = analyzer.analyze_code_structure(sample_apex)
        
        print(f"  • Lines of code: {analysis['lines_of_code']}")
        print(f"  • SOQL queries: {analysis['soql_queries']}")
        print(f"  • DML operations: {analysis['dml_operations']}")
        print(f"  • SOQL in loops: {analysis['soql_in_loops']}")
        print(f"  • DML in loops: {analysis['dml_in_loops']}")
        print(f"  • Nested loops: {analysis['nested_loops']}")
        print(f"  • Cyclomatic complexity: {analysis['cyclomatic_complexity']}")
        
        # Test performance simulator
        print("\n⚡ Testing ApexPerformanceSimulator...")
        
        simulator = ApexPerformanceSimulator()
        performance = simulator.simulate_performance(analysis, record_count=50)
        
        print(f"  • CPU time: {performance['cpu_time_ms']:.1f} ms")
        print(f"  • Memory usage: {performance['memory_usage_kb']:.1f} KB")
        print(f"  • Projected SOQL queries: {performance['projected_soql_queries']}")
        print(f"  • Projected DML operations: {performance['projected_dml_operations']}")
        print(f"  • Execution efficiency: {performance['execution_efficiency']:.3f}")
        print(f"  • Governor limit risk: {performance['governor_limit_risk']:.3f}")
        
        print("✅ Component testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Component testing failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Apex Optimization Evaluator Test Suite")
    print("=" * 60)
    
    # Test the main evaluation
    success1 = test_initial_program()
    
    # Test individual components  
    success2 = test_evaluator_components()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! The evaluator is working correctly.")
        print("\n🚀 You can now run the full evolution:")
        print("   python ../../openevolve-run.py initial_program.apex evaluator.py --config config.yaml")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 