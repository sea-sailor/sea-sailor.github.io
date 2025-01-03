---
title: "Resources"
search_hidden: true
show_bread_crumbs: false
show_code_copy_buttons: true
disable_share: true
hide_meta: true
header:
    background: "linear-gradient(to top, #6a11cb 0%, #2575fc 100%);"
---
## Links

In the following, we provide important links for you to refer to our opensource resources.


{{< button href="https://arxiv.org/abs/2404.03608" label="PAPER" external=true >}}
{{< button href="https://github.com/sail-sg/sailor2" label="GITHUB" external=true >}}
{{< button href="https://huggingface.co/collections/sail/sailor2-language-models-674d7c9e6b4dbbd9a869906b" label="HUGGING FACE" external=true >}}


## Quick Start

It is simple to use Sailor models through Hugging Face Transformers. Below is a demo usage for a quick start:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    'sail/Sailor2-20B-Chat',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained('sail/Sailor2-20B-Chat')
system_prompt= \
'You are an AI assistant named Sailor2, created by Sea AI Lab. \
As an AI assistant, you can answer questions in English, Chinese, and Southeast Asian languages \
such as Burmese, Cebuano, Ilocano, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tagalog, Thai, Vietnamese, and Waray. \
Your responses should be friendly, unbiased, informative, detailed, and faithful.'

prompt = "Beri saya pengenalan singkat tentang model bahasa besar."
# prompt = "Hãy cho tôi một giới thiệu ngắn gọn về mô hình ngôn ngữ lớn."
# prompt = "ให้ฉันแนะนำสั้น ๆ เกี่ยวกับโมเดลภาษาขนาดใหญ่"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
input_ids = model_inputs.input_ids.to(device)

generated_ids = model.generate(
    input_ids,
    max_new_tokens=512,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```