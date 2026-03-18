from torch.xpu import device
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    device="cuda:0",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

prompt = "What is the meaning of life?<|assistant|>"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

generation_output = model.generate(
    input_ids,
    max_new_tokens=100,)

print(tokenizer.decode(generation_output[0], skip_special_tokens=True))