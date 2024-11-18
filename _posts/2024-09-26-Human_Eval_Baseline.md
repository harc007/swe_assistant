# Introduction
In the previous [post](https://harc007.github.io/swe_assistant/2024/09/23/prompting_basic_algo.html), we saw one example of writing a prompt and generating python code. But, we can't decide to use a specific LLM based on one example right? Do we have a standard dataset of prompts and unit tests that we would like to run and evaluate various LLMs at their code generation capabilities? Answer, of course is yes we do have that. The most commonly used baseline dataset is called the human-eval dataset. In this post, we will deep dive into the structure of the dataset, how it is used, how the code generated is evaluated, what is the evaluation metric that is used and how are the LLMs performing on this dataset as of today. Uffff! Lots to cover. Lets jump right in!
# Human-Eval
Human-Eval dataset is a open dataset created by OpenAI that has 164 different Python programs with 6 tests for each program. Over the last couple of years, this dataset is the gold standard used by various organizations training their own LLMs to evaluate code generation capabilities of their LLMs. You can download this dataset from [here](https://github.com/openai/human-eval/tree/master/data) link. If you want ways to run this quickly on your LLM, you can find them [here](https://github.com/openai/human-eval). But if you are interested to see what an example looks like and deep dive a bit, continue reading. Let us now see what an example looks like.
# Example Human-Eval
Once you download the .gz file in the link above, you can use the below code to stream the examples.
```
from typing import Iterable, Dict
import gzip
import json
import os

HUMAN_EVAL_LOC = # YOUR-FILE-LOCATION-HERE
def stream_jsonl(filename:str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith('.gz'):
        with open(filename, 'rb') as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, 'r') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

human_eval_stream = stream_jsonl(HUMAN_EVAL_LOC)

for sample in human_eval_stream:
    print(sample)
```
Above code prints all the 164 examples. Lets see how one of them looks like.
```
{
    'task_id': 'HumanEval/3', 
    'prompt':  'from typing import List
                
                def below_zero(operations: List[int]) -> bool:
                    """You\'re given a list of deposit and withdrawal operations on a bank account that starts with
                       zero balance. Your task is to detect if at any point the balance of account falls below zero, 
                       and at that point function should return True. Otherwise it should return False. 

                    >>>below_zero([1, 2, 3])
                    False   
                    >>>below_zero([1, 2, -4, 5])
                    True
                    """
                ', 
    'entry_point': 'below_zero', 
    'canonical_solution': ' balance=0
                            for op in operations:
                                balance += op
                                if balance < 0:
                                    return True
                            return False
                           ', 
    'test': "METADATA = {'author':'jt', 'dataset':'test'}
             def check(candidate):
                 assert candidate([]) == False
                 assert candidate([1, 2, -3, 1, 2, -3]) == False
                 assert candidate([1, 2, -4, 5, 6]) == True
                 assert candidate([1, -1, 2, -2, 5, -5, , -4]) == False 
                 assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
                 assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True
            "
}
```
Ok, that is an interesting dictionary. But what do I do next? 
# How is this used?
Following are steps to be done from now:
1. The idea  is that we pass the 'prompt' from the dictionary to the LLM we want to evaluate
2. Get the code generated from the LLM. Hopefully, string generated is the code that needs to be put in under the function defined within 'prompt'
3.  Once that is done, we can execute the unit tests as defined within the 'test' key in above dictionary. 

Let us try doing that by ourselves using the OpenAI gpt-3.5 model.

```
from dotenv import load_dotenv
load_dotenv()
import openai

stop = ['def']
model = 'gpt-3.5-turbo-instruct-0914'
max_tokens = 512
temperature = 0.2
num_samples_per_task = 1
code_signature = sample['prompt'] # From the code above

response = openai.completions.create(
    model = model, 
    prompt = code_signature, 
    max_tokens = max_tokens, 
    temperature = temperature, 
    stop = stop
)
print(response.choices[0].text)
``` 
The prompt passed was not an entirely english description of the kind of code we wanted to generate like the one we gave in the previous post. Now, it is a pythonic function definition with docstring. Lets what the gpt-3.5 model generated.

```
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
```
That is exactly what we wanted. 
# How is this evaluated
Now lets find out how do we execute the generated code on the unit tests provided in the 'test' part of the human-eval dictionary.
```
generated_code = response.choices[0].text
# Adding the '#\n' in code below to see parts of the string concatenation
code_to_execute = sample['prompt'] + '#\n' + 
                    generated_code + '#\n' + 
                    sample['test'] + '#\n' +
                    f'check({sample["entry_point"]})'
result = []
try:
    exec(code_to_execute)
    result.append("passed")
except BaseException as e:
    result.append("failed: {e}")
```
Once you run this on all the 16 examples, you can count how many 'passed' and how any 'failed' from the 'result' variable above. If you really want to see what the string 'code_to_execute' looks like, check it out below. Honestly, I got a fair understanding only after looking at it.
```
from typing import List
                
def below_zero(operations: List[int]) -> bool:
    """You\'re given a list of deposit and withdrawal operations on a bank account that starts with
        zero balance. Your task is to detect if at any point the balance of account falls below zero, 
        and at that point function should return True. Otherwise it should return False. 

    >>>below_zero([1, 2, 3])
    False   
    >>>below_zero([1, 2, -4, 5])
    True
    """
#
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
#

METADATA = {'author':'jt', 'dataset':'test'}
def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, , -4]) == False 
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) ==True

check(below_zero)
```
# Evaluation Metric
The evaluation metric calculated above is pass@1. That is because we have generated one piece of code from the LLM and have identified how many are correct. This can be generalized to pass@k. This is where we generate multiple answers from the same LLM for the same human-eval problem and find out how many pass the unit test. The formula for pass@k is 1 - {C(n-c, k)/C(n, k)} where C is number of combinations, n is total number of code pieces generated, c is number of correct code pieces, k is the number of code generated per prompt. The above is valid when n-c > k. Else pass@k is 1.
# Human-Eval leaderboard
![alt text](/swe_assistant/docs/assets/he_lb_20240926.JPG "Leaderboard")
The above plot shows that pass@1 scores have breached the 99% mark by the O1-mini models. It was around 32% in 2021. This is incredible progress in 2-3 years.
# Conclusion
In this blog, we saw a baseline dataset that is used widely to report on code generation capabilities of LLMs. We saw how it is structured and how to use it on a LLM of our choice. We also saw the evaluation metric used to report numbers. The plot above also shows the tremendous empirical improvement in code generation capabilities for the Human-Eval kind of problem statements over the last 2-3 years. Which brings us to the next question. Given that we have hit 99%+ in Human-Eval, is this dataset of any significance moving forward? Until next time, sleep well. :)
