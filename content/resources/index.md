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
{{< button href="https://github.com/sail-sg/sailor-llm" label="GITHUB" external=true >}}
{{< button href="https://huggingface.co/collections/sail/sailor-65e19a749f978976f1959825" label="HUGGING FACE" external=true >}}


## Quick Start

It is simple to use Sailor models through Hugging Face Transformers. Below is a demo usage for a quick start:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    'sail/Sailor-14B-Chat',
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained('sail/Sailor-14B-Chat')
system_prompt= \
'You are an AI assistant named Sailor created by Sea AI Lab. \
As an AI assistant, you need to answer a series of questions next, which may include languages such as English, Chinese, Thai, Vietnamese, Indonesian, Malay, and so on. \
Your answer should be friendly, unbiased, faithful, informative and detailed.'

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