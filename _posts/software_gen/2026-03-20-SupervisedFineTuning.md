---
layout: post
title: "SupervisedFineTuning"
categories: Random Exps
---
# Introduction
We understood the DistilBERT model and its components in the last [blog](https://harc007.github.io/swe_assistant/random/exps/2026/02/28/distilBERT.html). In this blog, we will take a dataset, define a classification task and learn to finetune DistilBERT using one fine-tuning technique called SupervisedFineTuning.

# Dataset
Let's use the IMDB dataset from huggingface datasets library. 
~~~
from datasets import load_dataset
imdb = load_dataset("imdb")
imdb["test"][0]
~~~
Output of above statements will give you:
~~~
{'text': 'I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn\'t match the background, and painfully one-dimensional characters cannot be overcome with a \'sci-fi\' setting. (I\'m sure there are those of you out there who think Babylon 5 is good sci-fi TV. It\'s not. It\'s clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It\'s really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it\'s rubbish as they have to always say "Gene Roddenberry\'s Earth..." otherwise people would not continue watching. Roddenberry\'s ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.',
 'label': 0}
~~~
So, we now have a perfect little play dataset with text reviews of movies and a label that tells if the review was positive or negative.

# Why DistilBERT
As discussed in previous blog, DistilBERT has ~66M params which would mean it easily fits into the 16GB RAM machine that I have. It also means that finetuning the model doesn't take a long time running to days. The finetuning will probably run for a couple of hours on the Mac mini that I have.

# DistilBERT Tokenizer
The role of the tokenizer is to convert the words into tokens. For eg, "I love you" could be converted to "[456, 5676, 436]". Where, in the tokenizer mapping, "I" corresponds to the 456th token and so on for other words. Every model has its own tokenizer based on the corpus they were trained on.
~~~
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
~~~
The entire dataset is now converted to tokenized content and ready for the DistilBERT encoder-only transformer model to consume.

# DistilBERT model with head
We now have to load the DistilBERT model and add a head to it so that it always produces a binary output since we want to do binary classification.
~~~
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
~~~
We saw this in previous blog as well. We have now take the distilbert model, added a Binary classification head and given it a mapping of what the label means.

# SFT
Lets now finetune the model to optimize on accuracy. 
~~~
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
~~~
That's it! You're done! Go take nap as this runs for a couple of hours if you are running on a mac mini. You have defined the metrics to track which is accuracy in this case. You have mentioned tthe learning rate, batch size, number of epochs etc.

The output that I obtained was as follows:
~~~
TrainOutput(global_step=6250, training_loss=0.23097577880859374, metrics={'train_runtime': 18108.879, 'train_samples_per_second': 2.761, 'train_steps_per_second': 0.345, 'total_flos': 6243221615506944.0, 'train_loss': 0.23097577880859374, 'epoch': 2.0})
~~~
This leads to an accuracy of ~93%.

# Conclusion
You have now learned to finetune an entire model. What this means is that this finetuned model will have same architecture of a distilbert but potentially all its weights could be changed as a result of finetuning. But this model should ideally be more suited for your binary classification task than the entire DistilBERT model in itself. 

Is this the most efficient way of finetuning? This seems quite tedious. Yes there are better alternatives to this. Let's learn about them in the next blog. Until then, Adios!