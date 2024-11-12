# Introduction
In the previous post, we saw how popular HumanEval dataset has become as a benchmark for evaluating code generating capabilities of LLMs. But, we also saw that the market has started producing LLMs that easily have a pass@1 metric of over 99% which has rendered the HumanEval dataset redundant. What next then?
# The Next Generation of HumanEval
The reason why the pass@1 metric is so high on the HumanEval dataset is because the tasks are algorithm-oriented whereas the real software development often involves usage of diverse libraries and calling functions defined in the libraries. Enter BigCodeBench - the next generation of HumanEval. BigCodeBench contains 1140 function-level tasks to challenge LLMs to follow instructions and compose multiple function calls as tools from 139 libraries. Each task has 5-6 test cases.
# How to load dataset
Below is code to load dataset from huggingface datasets.
```
from datasets import load_dataset
bcb_data = load_dataset('bigcode/bigcodebench')
```
The dataset dictionary consists of 4 versions, we can just focus on latest version v0.1.2. Now, each of the tasks have a complete_prompt, instruct_prompt, test as the important keys. 
The complete_prompt is the code prompt where the LLM is expected to fill in the blanks with code. Whereas, the instruct_prompt is more of a conversational style English content that requests for the entire code to be written. The test is the test cases for that particular task. Lets look at one of the example in detail.
# Dataset detail
For the first task, the complete_prompt looks as shown below
```
#print(bcb_data['v0.1.2']['complete_prompt'][0])

import itertools
from random import shuffle

def task_func(numbers=list(range(1, 3))):
    """
    Calculates the average of the sums of absolute differences between each pair of consecutive numbers
    for all permutations of a given list. Each permutation is shuffled before calculating the differences.

    Args:
    - numbers (list): A list of numbers. Default is numbers from 1 to 10.

    Returns:
    float: The average of the sums of absolute differences for each shuffled permutation of the list

    Requirements:
    - itertools
    - random.shuffle

    Example:
    >>> result = task_func([1, 2, 3])
    >>> isinstance(result, float)
    True
    """
```
For the same task, lets take a look at the instruct prompt
```
#print(bcb_data['v0.1.2']['instruct_prompt'][0])

Calculate the average of the sums of absolute differences between each pair of consecutive numbers for all permutations of a given list. Each permutaion is shuffled before calculating the differences. Args: - numbers (list): A list of numbers. Default is numbers from 1 to 10. The function should output with:
    float: The average of the sums of absolute differences for each shuffled permutation of the list.
You should write self-contained code starting with:

import itertools
from random import shuffle
def task_func(numbers=list(range(1, 3))):
```
It is quite clear that the 2 prompts are asking the LLM to come up with a solution to the same problem. Let us now take a look at the test cases for this problem.
```
# print(bcb_data['v0.1.2']['test'][0])

import unittest
from unittest.mock import patch
from random import seed, shuffle
import itertools
class TestCases(unittest.TestCase):
    def test_default_numbers(self):
        # Test with default number range (1 to 10) to check that the result is a positive float.
        result = task_func()
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    def test_custom_list(self):
        # Test with custom list of small positive integers to ensure proper handling and positive result.
        result = task_func([1, 2, 3])
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    def test_negative_numbers(self):
        # Test with negative numbers to verify the function handles and returns a positive result.
        result = task_func([-3, -2, -1])
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    def test_single_element(self):
        # Test with single element list to confirm result is zero since no pairs exist.
        result = task_func([5])
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0)
    def test_empty_list(self):
        # Test with an empty list to ensure the function handles it gracefully and returns zero.
        result = task_func([])
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    def test_identical_elements(self):
        # Test with a list of identical elements to confirm that differences are zero and the average is zero.
        result = task_func([2, 2, 2])
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0)
    def test_mixed_numbers(self):
        # Test with a list of mixed positive and negative numbers to check correct average of differences.
        result = task_func([-10, 10, -5])
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    def test_specific_value_with_seed(self):
        # Set seed for reproducibility and check the computed value
        with patch('random.shuffle', side_effect=lambda x: seed(42) or shuffle(x)):
            result = task_func([1, 2, 3])
            self.assertAlmostEqual(result, 2.5, delta=0.5) # This expected value should be calculated beforehand
    def test_large_list_with_seed(self):
        # Set seed and test with a larger list for specific computed value
        with patch('random.shuffle', side_effect=lambda x: seed(99) or shuffle(x)):
            result = task_func(list(range(1, 11)))
            self.assertAlmostEqual(result, 33.0, delta=0.5) # This expected value should be calculated beforehand
    def test_random_behavior(self):
        # Test to ensure different seeds produce different outputs, demonstrating randomness
        with patch('random.shuffle', side_effect=lambda x: seed(1) or shuffle(x)):
            result1 = task_func([1, 2, 3])
        with patch('random.shuffle', side_effect=lambda x: seed(1) or shuffle(x)):
            result2 = task_func([1, 2, 4])
        self.assertNotEqual(result1, result2)
```
# Running above on LLM
Let's now try to run the above model on gpt3.5. Code to that is as shown below
```
import openai
model = 'gpt-3.5-turbo-instruct-0914'
max_tokens = 512
temperature = 0.2
num_samples_per_task = 1
code_signature = bcb_data['v0.1.2']['instruct_prompt'][0]
response = openai.completions.create(
    model = model, 
    prompt = code_signature, 
    max_tokens = max_tokens, 
    temperature = temperature
)
print(response.choices[0].text)
```
The output is the program as shown below
```
import itertools
from random import shuffle

def task_func(numbers=list(range(1, 11))):
    # Generate all permutations of the given list
    permutations = itertools.permutations(numbers)
    # Initialize a list to store the sums of absolute difference for each pemutation
    sums = []
    for permutation in permutations:
        # shuffle the permutation
        shuffle(permutation)
        # Initialize a variable to store the sum of absolute differences for this permutation
        sum_diff = 0
        # Loop through each pair of consecutive numbers in the shuffled permutation
        for i in range(len(permutation)-1):
            # Calculate the absolute difference between the two numbers
            diff = abs(permutation[i] - permutation[i+1])
            # Add the difference to the sum
            sum_diff += diff
        # Add the sum of absolute differences for this permutation to the list
        sums.append(sum_diff)
    # Calculate the average of the sums of absolute differences
    avg = sum(sums)/len(sums)
    # Return the average
    return avg
```
Looks like the correct code has been generated. Sound logic. But the above code throws an error. A very simple error. While shuffling the permutation, it needs to convert the 'permutation' to 'list(permutation)'. If that change is done, the code generated passes through all tests.
# BigCodeBench benchmarking
![alt text](/swe_assistant/docs/assets/bcb_lb_20241112.JPG "Leaderboard")
The above snapshot shows the leaderboard on the begcodebench benchmarking of LLMs and it can be seen that the best model is GPT4 with a pass@1 of 32.1%. There seems to be a lot of room for improvement in the tasks.
# Conclusion
In the single example, we saw, it was not a case of the LLM being unable to understand the problem or solve for it. It got it wrong in minutae like converting a object of type itertools to a list. The question is if that is the case across examples. If that is so, can we add guardrails to LLMs to make it a little better and help it avoid simple errors? We definitely now have a dataset where there is huge room to improve. That is the biggest takeaway from this post. We made huge ground today. Until next time, get some rest :)