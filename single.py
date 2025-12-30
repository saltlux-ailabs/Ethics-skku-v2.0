import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
from hook_utils import hook_subtract_with_weighted_sum, hook_subtract
import pdb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(42)

hook_config = {
    "scale": 50,
    "subtract_from": [9], #[31]
    "hook_timesteps": -1,
    "type": "toxic_probe",
    "datapath": None, # fill in
}

target = "single_fair_50_large_balance"

model_name = "gpt2-large"
# model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

input_path = None # fill-in
output_path_gen = None # fill-in
output_path_full = None # fill-in
batch_size = 32
max_new_tokens = 20


if torch.cuda.is_available():
    model.to("cuda")

# # 프롬프트 불러오기
prompts = []

with open(None) as f: #fill in
    data = [json.loads(json_i) for json_i in f.readlines()]
for prompt_i in data:
    prompts.append(prompt_i['prompt']['text'])

with open(output_path_gen, "w", encoding="utf-8") as gen_f, \
     open(output_path_full, "w", encoding="utf-8") as full_f:

    # 배치 단위로 생성
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        hook_config["attention_mask"] = attention_mask
        hook_config["prompt_input_ids"] = input_ids
        
        model, hooks = hook_subtract(model, hook_config)

        batch_gen = []

        for j in range(5):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                min_new_tokens=3,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id
            )

            for h in hooks:
                h.remove()

            generated_tokens = outputs[:, input_ids.shape[1]:]
            decoded_generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            batch_gen.append(decoded_generated)

        for idx in range(len(decoded_generated)):
            for j in range(5):
                gen_f.write(json.dumps(batch_gen[j][idx].strip(), ensure_ascii=False) + "\n")    

print(f"✅ 생성 완료 및 저장됨:\n- 생성된 텍스트: {output_path_gen}\n- 프롬프트와 생성된 텍스트: {output_path_full}")