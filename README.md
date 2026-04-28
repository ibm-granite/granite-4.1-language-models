<p align="center">
  <img src="figures/granite-4_1-language-models-3x-v1.png" />
</p>

<p align="center">
  :hugs: <a href="https://huggingface.co/collections/ibm-granite/granite-41-language-models">HuggingFace Collection</a>&nbsp | :hugs: <a href="https://huggingface.co/blog/ibm-granite/granite-4-1">HuggingFace Technical Blog</a> | :speech_balloon: <a href="https://github.com/orgs/ibm-granite/discussions">Discussions Page</a>&nbsp | 📘 <a href="https://www.ibm.com/granite/docs/">IBM Granite Docs </a>
<br>

---
## Overview
Granite 4.1 language models are a family of state-of-the-art open foundation models featuring dense decoder-only architectures in three sizes — 3B, 8B, and 30B. They natively support multilingual capabilities, a wide range of coding tasks, retrieval-augmented generation (RAG), tool usage, and structured JSON output.

Our models are trained from scratch on approximately 15 trillion tokens through a five-phase strategy designed to progressively refine data quality and model capabilities. The first two phases cover pre-training proper, before transitioning into mid-training in Phases 3 and 4 with high-quality data annealing. The fifth and final phase performs long-context extension, scaling the context window up to 512K tokens through a staged process.

All models are publicly released under the Apache 2.0 license, allowing free use for both research and commercial purposes. The data curation and training processes were specifically designed for enterprise scenarios and customization, incorporating governance, risk, and compliance (GRC) evaluations alongside IBM's standard data clearance and document quality review procedures.

We provide both base models (checkpoints after pre-training) and instruct models (checkpoints fine-tuned for dialogue, instruction following, helpfulness, and safety).

<!-- Comprehensive evaluation results for all model variants, as well as other relevant information will be available in Granite 4.1 Language Models technical report. -->

## How to Use our Models?
To use any of our models, pick an appropriate `model_path` from:
1. `ibm-granite/granite-4.1-3b-base`
2. `ibm-granite/granite-4.1-3b`
3. `ibm-granite/granite-4.1-8b-base`
4. `ibm-granite/granite-4.1-8b`
5. `ibm-granite/granite-4.1-30b-base`
6. `ibm-granite/granite-4.1-30b`

## Inference Examples

### Basic Inference
This is a simple example of how to use Granite-4.1-3B model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "auto"
model_path = "ibm-granite/granite-4.1-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
# change input text as desired
chat = [
    { "role": "user", "content": "What is the name of the durable rock known for being one of the hardest natural building stones?"},

]

chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=150)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)
```

### Tool-calling capabilities for AI agents
Agentic tool-calling is shaping the future of AI agents, enabling seamless integration of powerful back-end systems into agent-driven workflows. These trajectories often involve multiple tool calls, handling execution responses, and multi-turn user interactions. While agent frameworks orchestrate long-horizon tasks, LLMs must provide the foundation — including standard tool formats, robust tool-call handling (even in edge cases), and support for feeding back execution results. 

The following code example demonstrates how Granite 4.1’s tool-calling capabilities address these needs. In the first user query, the model successfully generates the appropriate tool call because it has access to the necessary tools. In contrast, it produces an apology message for the second query, as the required tooling is unavailable. Since this example does not use an agent framework, tool execution is simulated.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
# model_path = ""
model_path = "ibm-granite/granite-4.1-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

chat=[
    {"role": "user", "content": "I'm looking to buy a used truck for my construction work, but I want to make sure it's legitimate. The seller provided the VIN: 1FMXK92W8YPA12345 and said it's registered in Georgia. Can you verify if the VIN is valid and matches a registered vehicle?"},
    {"role": "assistant", 
        "content": "", 
        "tool_calls": [
            {
                "function": {
                    "name": "check_valid_vin",
                    "arguments": {"vin": "1FMXK92W8YPA12345"}
                }
            }
        ]
    },
    {"role": "tool",  "content": "{\"valid\": true, \"vin_details\": {\"make\": \"Ford\", \"model\": \"F-150\", \"year\": 2020, \"vehicle_type\": \"Truck\", \"registration_status\": \"Active\", \"registration_state\": \"GA\", \"odometer\": 82345, \"title_status\": \"Clear\", \"lienholder\": null, \"recall_history\": \"No active recalls\"}, \"notes\": \"VIN is valid and registered in Georgia. PPSR lien check complete - no security interests found. License plate verification requires separate DMV lookup which is not currently available through this tool.\"}"},
    {"role": "user", "content": "I'm also considering purchasing a new Ford F-150 from an official dealership in Texas. Could you provide a cost estimate for this type of truck in that state?"},  
]

tools = [
  {
    "type": "function",
    "function": {
      "name": "check_valid_registration",
      "description": "Verifies whether a vehicle registration number is valid for a specific state and returns detailed information about the registered vehicle if valid. Use this function to validate vehicle registration status and obtain ownership/vehicle data.",
      "parameters": {
        "type": "object",
        "properties": {
          "reg": {
            "type": "string",
            "description": "Vehicle registration number in standard format (e.g., ABC123 or XYZ-7890)"
          },
          "state": {
            "type": "string",
            "description": "Two-letter state abbreviation where the vehicle is registered (e.g., CA for California, NSW for New South Wales, or TX for Texas)"
          }
        },
        "required": ["reg", "state"],
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "check_valid_vin",
      "description": "Verifies if a vehicle identification number (VIN) corresponds to a registered vehicle in official records. Returns comprehensive vehicle details including make, model, year, registration status, and ownership information if valid.",
      "parameters": {
        "type": "object",
        "properties": {
          "vin": {
            "type": "string",
            "description": "The 17-character Vehicle Identification Number to validate. Must follow standard VIN format (uppercase alphanumeric characters, no spaces or special characters). Case-insensitive validation performed internally."
          }
        },
        "required": ["vin"],
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ppsr_lookup_by_vin",
      "description": "Performs a PPSR (Personal Property Securities Register) lookup for a vehicle using its VIN. Returns search results including security interests, ownership status, and an official PDF certificate URL. Use this to verify vehicle history or security claims.",
      "parameters": {
        "type": "object",
        "properties": {
          "vin": {
            "type": "string",
            "description": "17-character alphanumeric vehicle identification number (ISO 3779 standard). Case-insensitive. Example: '1HGCM82633A123456'"
          }
        },
        "required": ["vin"]
      }
    }
  },
]
    
chat = tokenizer.apply_chat_template(chat,tokenize=False, add_generation_prompt=True, tools=tools)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=1000)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output[0])
```

## JSON as Output
Granite natively supports structured JSON output, a capability that proves useful in many real-world scenarios. Below, we demonstrate how it can be applied to parse natural language IT support tickets.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model_path = "ibm-granite/granite-4.1-3b" 

tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

chat = [
  {
    "role": "system",
    "content": "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{\"title\":\"ITSupportTicket\",\"type\":\"object\",\"properties\":{\"ticketID\":{\"type\":\"string\"},\"requester\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"email\":{\"type\":\"string\",\"format\":\"email\"}},\"required\":[\"name\",\"email\"]},\"category\":{\"type\":\"string\",\"enum\":[\"Access\",\"Hardware\",\"Software\",\"Network\",\"Other\"]},\"priority\":{\"type\":\"string\",\"enum\":[\"Low\",\"Medium\",\"High\",\"Critical\"]},\"description\":{\"type\":\"string\"},\"reportedAt\":{\"type\":\"string\",\"format\":\"date-time\"}},\"required\":[\"ticketID\",\"requester\",\"category\",\"priority\",\"description\",\"reportedAt\"]}\n</schema>\n"
  },
  {
    "role": "user",
    "content": "Please breakdown the follwing IT ticket's content and classify it for me.\n# This is the ticket:\nI can't access the VPN from home—keeps timing out after authentication. Please mark this as High priority. I'm Jordan Lee, jordan.lee@acme.co. Create the ticket as TCK-10944. I noticed the issue today around 2025-10-01T08:35:00Z."
  }
]

chat = tokenizer.apply_chat_template(chat,tokenize=False, add_generation_prompt=True)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=200)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output[0])
```

## Fill-in-the-middle (FIM) Code Completion
The FIM (Fill-in-the-Middle) code completion capability is highly valuable for software developers, as it enables models to intelligently generate missing code segments within existing functions. The example below demonstrates how it can be used to complete a function that processes user data and produces a summary.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
# model_path = ""
model_path = "ibm-granite/granite-4.1-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

prompt = """<|fim_prefix|>
def summarize_users(users):
    \"\"\"
    Given a list of user dictionaries with 'name' and 'age',
    return a summary with the average age and a list of names.
    \"\"\"
    summary = {}
<|fim_suffix|>
    return summary
<|fim_middle|>
"""

chat = [
    { "role": "user", "content": prompt},
]

chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output[0])
```

## How to Download our Models?
The model of choice (`ibm-granite/granite-4.1-8b` in this example) can be cloned using:
```shell
git clone https://https://huggingface.co/ibm-granite/granite-4.1-8b
```

## How to Contribute to this Project?
Plese check our [Guidelines](/CONTRIBUTING.md) and [Code of Conduct](/CODE_OF_CONDUCT.md) to contribute to our project.

## Model Cards
The model cards for each model variant are available in their respective HuggingFace repository. Please visit our collection [here](https://huggingface.co/collections/ibm-granite/granite-41-language-models).

## License 
All Granite 4.1 Language Models are distributed under [Apache 2.0](./LICENSE) license.

## Disclosures
Please find disclosures information [here](https://github.com/ibm-granite/granite-1.0-language-models/tree/main/disclosures).

## Would you like to provide feedback?
Please let us know your comments about our family of language models by visiting our [collection](https://huggingface.co/collections/ibm-granite/granite-41-language-models). Select the repository of the model you would like to provide feedback about. Then, go to *Community* tab, and click on *New discussion*. Alternatively, you can also post any questions/comments on our [github discussions page](https://github.com/orgs/ibm-granite/discussions).

## Citation
If you find granite models useful, please cite our work as follows:

```
@misc{granite2026,
  author       = {{IBM Research}},
  title        = {Granite 4.1 Language Models},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/blog/ibm-granite/granit-4-1}},
  note         = {Accessed: 2026-04-28}
}
```


