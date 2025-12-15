#!/usr/bin/env python3
import json
import argparse
import sys
from llm_client import query_llm

def create_prompt(scenario, items, outcome, question, statement):
    """
    Create a prompt for scalar implicature evaluation.
    
    Args:
        scenario: The setup (e.g., "The monkey loves eating yummy stuff.")
        items: Available items (e.g., "There's a banana, a cake, an orange and a biscuit")
        outcome: What actually happened (e.g., "The monkey ate the orange and the biscuit")
        question: The experimenter's question (e.g., "So, what did the monkey eat?")
        statement: The statement to evaluate (e.g., "The monkey ate the biscuit")
    
    Returns:
        str: Complete prompt for the LLM
    """
    prompt = f"""Read this scenario carefully:

{scenario}
{items}
{outcome}

{question}
Answer: "{statement}"

Is this answer correct? Respond with only "yes" or "no"."""
    
    return prompt

def evaluate_response(response_text, expected_answer):
    """
    Check if the LLM's response matches the expected answer.
    
    Args:
        response_text: The LLM's response
        expected_answer: Expected "yes" or "no"
    
    Returns:
        bool: True if correct, False otherwise
    """
    response_lower = response_text.lower().strip()
    expected_lower = expected_answer.lower().strip()
    
    # Handle various response formats
    if expected_lower in response_lower:
        return True
    
    # Check for common variations
    if expected_lower == "no" and any(word in response_lower for word in ["incorrect", "false", "wrong"]):
        return True
    if expected_lower == "yes" and any(word in response_lower for word in ["correct", "true", "right"]):
        return True
        
    return False

def extract_yes_no_confidence(logprobs_data, response_text):
    """
    Extract confidence scores for yes/no responses from logprobs.
    
    Args:
        logprobs_data: Logprobs data from the API
        response_text: The actual response text
    
    Returns:
        dict: Confidence information including probabilities for yes/no
    """
    import numpy as np
    
    if not logprobs_data or not logprobs_data.content:
        return None
    
    # Look for the first token that contains yes/no
    for token_data in logprobs_data.content:
        token = token_data.token.lower().strip()
        
        if 'yes' in token or 'no' in token:
            confidence_data = {
                'chosen_token': token_data.token,
                'chosen_logprob': token_data.logprob,
                'chosen_probability': np.exp(token_data.logprob),
                'alternatives': []
            }
            
            # Add top alternatives
            if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                for alt in token_data.top_logprobs:
                    confidence_data['alternatives'].append({
                        'token': alt.token,
                        'logprob': alt.logprob,
                        'probability': np.exp(alt.logprob)
                    })
            
            return confidence_data
    
    return None

def run_single_test(test_case, model, temperature=0.0, use_logprobs=False):
    """
    Run a single scalar implicature test.
    
    Args:
        test_case: Dictionary with test case data
        model: Model to use
        temperature: Sampling temperature
        use_logprobs: Whether to return logprobs data
    
    Returns:
        dict: Results including correctness and optionally logprobs
    """
    prompt = create_prompt(
        test_case['scenario'],
        test_case['items'], 
        test_case['outcome'],
        test_case['question'],
        test_case['statement']
    )
    
    try:
        if use_logprobs:
            response_data = query_llm(prompt, model, return_usage=True, logprobs=True, temperature=temperature)
            response = response_data['text']
            logprobs_data = response_data.get('logprobs')
        else:
            response = query_llm(prompt, model, temperature=temperature)
            logprobs_data = None
        
        is_correct = evaluate_response(response, test_case['expected'])
        
        result = {
            'prompt': prompt,
            'response': response,
            'expected': test_case['expected'],
            'correct': is_correct,
            'category': test_case['category']
        }
        
        if use_logprobs and logprobs_data:
            result['logprobs'] = extract_yes_no_confidence(logprobs_data, response)
        
        return result
    except Exception as e:
        print(f"Error processing test case: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate scalar implicature understanding')
    parser.add_argument('test_file', help='JSON file with test cases')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--logprobs', action='store_true', help='Include logprobs and confidence analysis')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load test cases
    try:
        with open(args.test_file, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Test file not found: {args.test_file}")
        sys.exit(1)
    
    results = []
    correct_by_category = {'true': 0, 'false': 0, 'underinformative': 0}
    total_by_category = {'true': 0, 'false': 0, 'underinformative': 0}
    
    print(f"Running {len(test_cases)} scalar implicature tests with {args.model}...")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}/{len(test_cases)}: {test_case['category']}")
        
        result = run_single_test(test_case, args.model, args.temperature, args.logprobs)
        if result:
            results.append(result)
            
            category = result['category']
            total_by_category[category] += 1
            if result['correct']:
                correct_by_category[category] += 1
            
            status = "✓" if result['correct'] else "✗"
            confidence_str = ""
            if args.logprobs and 'logprobs' in result and result['logprobs']:
                prob = result['logprobs']['chosen_probability']
                confidence_str = f" (confidence: {prob:.3f})"
            
            print(f"  {status} Expected: {result['expected']}, Got: {result['response'][:50]}...{confidence_str}")
        
        print()
    
    # Calculate statistics
    total_correct = sum(correct_by_category.values())
    total_tests = sum(total_by_category.values())
    
    print("="*80)
    print("SCALAR IMPLICATURE EVALUATION RESULTS")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Overall Accuracy: {total_correct/total_tests:.3f} ({total_correct}/{total_tests})")
    print()
    
    for category in ['true', 'false', 'underinformative']:
        if total_by_category[category] > 0:
            accuracy = correct_by_category[category] / total_by_category[category]
            print(f"{category.capitalize()} statements: {accuracy:.3f} ({correct_by_category[category]}/{total_by_category[category]})")
            
            if args.logprobs:
                # Calculate average confidence for this category
                category_results = [r for r in results if r['category'] == category and 'logprobs' in r and r['logprobs']]
                if category_results:
                    avg_confidence = sum(r['logprobs']['chosen_probability'] for r in category_results) / len(category_results)
                    print(f"  Average confidence: {avg_confidence:.3f}")
    
    if args.logprobs:
        print(f"\nOverall average confidence: {sum(r['logprobs']['chosen_probability'] for r in results if 'logprobs' in r and r['logprobs']) / len([r for r in results if 'logprobs' in r and r['logprobs']]):.3f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'model': args.model,
                'temperature': args.temperature,
                'total_accuracy': total_correct/total_tests,
                'accuracy_by_category': {cat: correct_by_category[cat]/total_by_category[cat] 
                                       for cat in correct_by_category if total_by_category[cat] > 0},
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()