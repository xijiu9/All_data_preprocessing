import torch
import numpy as np
from transformers import BioGptTokenizer, BioGptForCausalLM

# Define the input sentence and object
input_text = "guizhi-fuling prevents diseases such as"
object_text = " Dysmenorrhea"

# Load the GPT model and tokenizer
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

max_length = 30
input_ids = []
input_ids.extend(tokenizer.encode(input_text)[1:])

object_ids = []
object_ids.extend(tokenizer.encode(object_text)[1:])

probability = 1
print(tokenizer.decode(input_ids))
for i in range(len(object_ids)):
    inputs = {"input_ids": torch.tensor([input_ids])}
    outputs = model(**inputs)
    logits = outputs.logits
    last_token_id = int(object_ids[i])
    # last_token_id = int(np.argmax(logits[0][-1].detach().numpy()))

    last_token = tokenizer.convert_ids_to_tokens(last_token_id)
    last_prob = torch.softmax(logits[0, -1], dim=-1)[last_token_id].item()
    probability *= last_prob

    if last_token == tokenizer.eos_token:  # Check if the last generated token is an end-of-sequence token
        break
    input_ids.append(last_token_id)
    print(tokenizer.decode(input_ids), probability)
# 你说过哪些不合时宜的大实话？我们学校的一个女生，在我们学校的一个

import IPython
IPython.embed()

# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# # Define the input sentence and object
# sentence = "I want to eat"
# object_name = "dumpling"
#
# # Load the GPT model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
# # Tokenize the object
# object_tokens = tokenizer.tokenize(object_name)
#
# # Initialize the input text
# input_text = sentence
#
# # Predict each token of the object sequentially
# for token in object_tokens:
#     # Combine the input text and token
#     input_text += " " + token
#
#     # Convert the input text to input IDs
#     input_ids = tokenizer.encode(input_text, return_tensors="pt")
#
#     # Generate predictions for the next token
#     with torch.no_grad():
#         outputs = model.generate(input_ids)
#
#     # Decode the generated output and get the next predicted token
#     predicted_token = tokenizer.decode(outputs[:, -1])
#
#     # Print the predicted token
#     print(f"Token: {predicted_token}")
#
# # Output the final predicted tokens
# predicted_tokens = tokenizer.tokenize(object_name)
# print("Predicted Tokens:", predicted_tokens)
