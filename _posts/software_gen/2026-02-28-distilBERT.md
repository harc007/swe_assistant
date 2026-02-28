---
layout: post
title: "DistilBERT"
categories: Random Exps
---
# Introduction
Lets now get rolling with using huggingface, loading a open source LLM. We will start with a really simple model - DistilBERT. If you really want to understand this further, you need to have a fair understanding of what Transformer architecture is. To read the best blog that I have seen for that, click [here](https://jalammar.github.io/illustrated-transformer/)

# DistilBERT
Before we get into DistilBERT, lets understand BERT. BERT stands for Bidirectional Encoder Representations from Transformers. It is a Google developed AI model that revolutionized NLP by reading text bidirectionally. This was probably the first transformer model that gained massive popularity and was adopted widely. BERT was a encoder only model. So, BERT was really popular for providing semantic understanding of words/sentences when converted to their embeddings. For eg, it would provide different embeddings for 'python' if the word was used in a programming language context or in the context of a reptile.
DistilBERT models are smaller, 40% lighter, and 60% faster versions of BERT, designed for efficiency while retaining 97% of its performance.

Here are the main types and variations of DistilBERT models available on huggingFace:
1. distilbert-base-uncased - The standard model trained on English text, where all text is lowercased before tokenization. It is ideal for general-purpose NLP tasks like classification.
2. distilbert-base-cased - (Less common) A version that retains capitalization, useful for tasks where case matters (e.g., Named Entity Recognition).
3. distilbert-base-multilingual-cased - (DistilmBERT) A multilingual version trained on Wikipedia in 104 different languages, supporting cross-lingual tasks.
4. Task-Specific Fine-tuned Models - While the above are base models, DistilBERT is frequently fine-tuned for specific tasks via the Hugging Face library, such as:
  a. distilbert-base-uncased-finetuned-sst-2-english: Optimized for sentiment analysis.
  b. DistilBertForQuestionAnswering: Specifically designed for QA tasks.
  c. DistilBertForTokenClassification: Used for NER or POS tagging.

# Load DistilBERT
Let's focus on the standard model which is the distilbert-base-uncased. This model has 66 million parameters. In terms of LLM sizes, estimated to be running into trillions of parameters as of today, the distilbert is really a small model. For estimating what it takes to load the model in your RAM, it's good to remember that a 1 billion parameter model at full precision (float-32) takes about 4GB. So a 66M model will be ~250MB to load. For inference, you need roughly about 25% extra RAM/VRAM for this model. This is because of something called KV cache which is used for faster inference. We wil not get into it today. Similarly if we were to train this model which is our goal in subsequent posts, we will need about 5-6x the RAM/VRAM to load the model. Remember that all these are rule of thumb extrapolations to get an idea of RAM usage for a model and not precise numbers. Going with this extrapolation, we see that a local machine with about 16G RAM should be sufficient to load, run inference on and train this model. Let's look at the code to load the model - 
~~~
from dotenv import load_dotenv
load_dotenv()
~~~
This is to set any environment variables and in our case the 'HF_TOKEN' so that model downloads are not throttled. If you have limited GPU memory, one can also set the 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' to 0 to allow machine to utilise the entire GPU memory for the task at hand. This is optional though.
~~~
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
~~~
This is to download the tokenizer of the model. Every model has its own tokenizer artifact. So, we need to pass text through the tokenizer, and the output can be passed to model. Lets now download the model.
~~~
from transformers import AutoModelForSequenceClassification

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
~~~
Let's not worry too much about the id2label and label2id now. But you can see that we have downloaded the distilber-base-uncased model and have added a head on top of the model to do classification. Huggingface through their transformers package allow us to do this easily. We specify that we want to load the distilbert model through the AutoModelForSequenceClassification and specify the num of labels. This automatically creates the distilbert model and adds a head to do binary classification.

# Conclusion
Whoopie! This was not a byte sized post that I was hoping to post. Now that we have loaded the model, lets dive deeper in next post. Until then, take deep breaths!