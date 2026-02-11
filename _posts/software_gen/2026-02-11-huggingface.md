---
layout: post
title: "Prompting"
categories: Random Exps
---
# Introduction
While working in a big corporate and in the right team gives you the perks of access to the latest and greatest paid tech tools including access to the best LLMs out there, it also made me not aware of what were truly possible/bottlenecks with open source. I decided to take the plunge and try it out myself.

# HuggingFace
For those who are not aware, HuggingFace is the 'home' of open-source AI/ML. This is the company that is host to and provides access to most open-source models for anyone. Started as a chatbot company, HuggingFace pivoted into the role of democratizing AI tech by offering tools and models that allows practitioners at various levels to tap into the potential of transformers without needs deep expertise on them.

# What does HuggingFace offer to a Pythonista
One can look at the models available on huggingface by opening their homepage and clicking on 'Models' on top. ![alt text](/swe_assistant/docs/assets/hf_models.jpg "HF models")
One can then filter for the task at hand, parameter size etc, as shown in the snapshot above. The next important consideration is whether one wants to use the HF inference API or download the model locally and use it.
There are pros and cons to both. The pros to downloading the model is it offers maximum privacy as your data does not go outside the scope of your system. You can also finetune the downloaded model to customize for specific needs. But it comes at t he cost of high hardware requirements, setup and maintenance. LLM inference hardware requirements scale with parameter count and precision, demanding roughly 2GB VRAM per 1B parameters for 16-bit, or <1GB for 4-bit quantization. A 7B-8B model requires ~16GB-24GB+ VRAM (e.g., RTX 3090/4090), while 70B+ models need multi-GPU setups (e.g., 2-8x H100/A100) or high-memory systems (e.g., 128GB+ RAM for CPU). 
If you want to try the inference API, you can filter for models that are hosted by an inference provider by clicking on 'Inference Available' on top-right. 
Once you have applied all filters, you can choose the most relevant model by factors like number of parameters, downloaded count, liked count, last updated date etc. This is where we get the code to use the model.

# How to use the chosen model
Once you click on the model, you will find a 'Use the model' dropdown on the right as shown in image below. ![alt text](/swe_assistant/docs/assets/hf_model_code.jpg "HF model code")
Click on it and choose how you want to use the model. Like discussed earlier, one has 2 options. Either to download the model locally and use it via huggingface transformers library. In which case one can click on 'transformers' to get the code. The other option is to use the HF inference API. In that case, one can click on 'Inference Providers' to get the code.

# HF Inference Token
For certain cases, you might need a inference api token. It is quite easy to generate your own token. Create your own profile in HuggingFace and click on 'Access Tokens' from your profile and follow steps to create your own access token. There are rate limits to usage. Use as per the your means.

# Conclusion
We now have the ability to use any open-source model hosted on HuggingFace. Let's try more experiments using this. Good luck. Eat healthy!