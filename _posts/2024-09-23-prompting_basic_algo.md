# Introduction
Lets take a simple example of generating a Python function that can accept a list of numbers and a threshold float value as inputs. It must return boolean True if any 2 numbers in list are closer than the threshold value. If not, it should return a boolean False.

Rather simple. Lets start there.
# Code that works
```
from typing import List
import numpy as np
import itertools

def check_threshold_in_list(numbers:List, threshold:float)->bool:
    """
    From a list of numbers, find if difference between any two numbers 
    in the list is less than the threshold. Return True if it is the case
    Else, return False
    """
    combos = itertools.permutations(numbers, len(numbers))
    for combo in combos:
        diffs = np.array(combo) - np.array(numbers)
        if any([diff!=0 and abs(diff)<threshold for diff in diffs]):
            return True
    return False

>>>l = [1, 1.5, 1.8]
>>>check_threshold_in_list(l, 0.4)
True
>>>check_threshold_in_list(l, 0.2)
False
```
The above is a reasonably decent solution to the problem that I quickly wrote and tested as shown above. The complexity is O(n) where n is number of elements in the input list.
Let's see what some LLM's can do.
# Entering the LLM world
The first and most important way to start this is to write a prompt. Let's start with a simple but well-written prompt.
```
prompt = """generate python code that accepts a list of numbers and a threshold float value as input. 
            It returns boolean True if any 2 numbers in list are closer than the threshold value. 
            If not it returns a boolean False"""
```
Hope you will agree that the above is a decent prompt that explains what I want exactly from the LLM. Now lets try this on a openai gpt 3.5 model.
```
import openai
from dotenv import load_dotenv
load_dotenv()

model = 'gpt-3.5-turbo-instruct-0914'
max_tokens = 512
temperature = 0.2

response = openai.completions.create(
    model = model, 
    prompt = prompt, 
    max_tokens = max_tokens, 
    temperature = temperature    
)

print(response.choices[0].text)
```
The printed string looks like this:
```
def check_threshold(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

Not bad at all. The complexity is O(n^2) which is not the most efficient way of writing this. But still, an accurate answer.
# Other models
I tried the same using Mistral-7B-v0.1 model which is a open source model and got the same result. I also tried the same on models like starcoder2 on huggingface and got empty string as the response. I'm still trying to figure out what went wrong there when the starcoder2 model is supposed to be good at generating code.
# Conclusion
Modern day LLMs seem to be extremely good at understanding simple function use-cases and writing basic code. Simple prompts still doesn't produce efficient code. But nevertheless, there was no fault in the logic generated. Until next time, get some sleep :)