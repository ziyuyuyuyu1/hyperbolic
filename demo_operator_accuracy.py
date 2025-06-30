#!/usr/bin/env python3
"""
Demo script to show operator-specific accuracy calculation and display.
This simulates what the evaluation script outputs for operator accuracy.
"""

import json
from collections import defaultdict

def demo_operator_accuracy():
    """Demonstrate operator-specific accuracy calculation."""
    
    # Simulate some test results (this would come from actual model evaluation)
    sample_results = [
        {"operator": "+", "is_correct": True, "expression": "2 + 3", "expected": "5", "predicted": "5"},
        {"operator": "+", "is_correct": True, "expression": "10 + 20", "expected": "30", "predicted": "30"},
        {"operator": "+", "is_correct": False, "expression": "15 + 25", "expected": "40", "predicted": "35"},
        
        {"operator": "-", "is_correct": True, "expression": "10 - 3", "expected": "7", "predicted": "7"},
        {"operator": "-", "is_correct": False, "expression": "20 - 8", "expected": "12", "predicted": "13"},
        {"operator": "-", "is_correct": True, "expression": "50 - 10", "expected": "40", "predicted": "40"},
        
        {"operator": "*", "is_correct": True, "expression": "3 * 4", "expected": "12", "predicted": "12"},
        {"operator": "*", "is_correct": False, "expression": "5 * 6", "expected": "30", "predicted": "25"},
        {"operator": "*", "is_correct": False, "expression": "7 * 8", "expected": "56", "predicted": "54"},
        
        {"operator": "/", "is_correct": True, "expression": "20 / 4", "expected": "5", "predicted": "5"},
        {"operator": "/", "is_correct": False, "expression": "30 / 6", "expected": "5", "predicted": "6"},
        {"operator": "/", "is_correct": True, "expression": "40 / 8", "expected": "5", "predicted": "5"},
    ]
    
    # Calculate operator-specific metrics
    operator_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in sample_results:
        op = result["operator"]
        operator_metrics[op]["total"] += 1
        if result["is_correct"]:
            operator_metrics[op]["correct"] += 1
    
    # Calculate accuracy for each operator
    for op in operator_metrics:
        metrics = operator_metrics[op]
        metrics["accuracy"] = metrics["correct"] / metrics["total"]
    
    # Display results
    print("="*60)
    print("OPERATOR-SPECIFIC ACCURACY DEMONSTRATION")
    print("="*60)
    
    print(f"\nOverall Results:")
    total_correct = sum(metrics["correct"] for metrics in operator_metrics.values())
    total_samples = sum(metrics["total"] for metrics in operator_metrics.values())
    overall_accuracy = total_correct / total_samples
    print(f"  Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    
    print(f"\n" + "="*40)
    print("OPERATOR-SPECIFIC ACCURACY")
    print("="*40)
    
    # Sort operators by accuracy
    sorted_operators = sorted(operator_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Operator':<8} {'Accuracy':<12} {'Correct/Total':<15} {'Success Rate':<15}")
    print("-" * 55)
    
    for op, metrics in sorted_operators:
        accuracy = metrics["accuracy"]
        correct = metrics["correct"]
        total = metrics["total"]
        success_rate = accuracy * 100
        
        print(f"{op:<8} {accuracy:<12.4f} {correct:<7}/{total:<7} {success_rate:<15.1f}%")
    
    print(f"\nDetailed Breakdown:")
    for op, metrics in sorted_operators:
        print(f"\n{op} Operations:")
        print(f"  • Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"  • Success Rate: {metrics['accuracy']*100:.1f}%")
        
        # Show examples for this operator
        op_results = [r for r in sample_results if r["operator"] == op]
        print(f"  • Examples:")
        for result in op_results:
            status = "✓" if result["is_correct"] else "✗"
            print(f"    {status} {result['expression']} = {result['expected']} (predicted: {result['predicted']})")
    
    print(f"\n" + "="*40)
    print("OPERATOR RANKING (by accuracy)")
    print("="*40)
    for i, (op, metrics) in enumerate(sorted_operators, 1):
        print(f"{i}. {op}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

if __name__ == "__main__":
    demo_operator_accuracy() 