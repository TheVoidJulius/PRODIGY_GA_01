from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_PATH = "../output/gpt2-text-gen"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

prompt = "Artificial Intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    inputs["input_ids"],
    max_length=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))

