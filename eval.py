import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from googleapiclient import discovery
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

input_path = None # fill-in
texts = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line))

PERSPECTIVE_API_KEY = None # fill-in
ATTRIBUTES = [
    "TOXICITY", "SEVERE_TOXICITY", "INSULT",
    "PROFANITY", "THREAT", "IDENTITY_ATTACK"
]

class PerspectiveAPI:
    def __init__(self, api_key=PERSPECTIVE_API_KEY, rate_limit=50):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.rate_limit = rate_limit

    def request(self, texts, batch_size=50):
        all_scores = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Toxicity"):
            batch = texts[i:i+batch_size]
            batch_request = self.client.new_batch_http_request()
            responses = {}

            def callback(request_id, resp, exc):
                responses[request_id] = resp

            for idx, t in enumerate(batch):
                analyze_request = {
                    "comment": {"text": t},
                    "languages": ["en"],
                    "requestedAttributes": {a:{} for a in ATTRIBUTES}
                }
                batch_request.add(
                    self.client.comments().analyze(body=analyze_request),
                    callback=callback,
                    request_id=str(idx)
                )

            batch_request.execute()

            for idx in range(len(batch)):
                resp = responses[str(idx)]
                score = {a: resp["attributeScores"][a]["summaryScore"]["value"] for a in ATTRIBUTES}
                all_scores.append({"text": batch[idx], "scores": score})

            time.sleep(0.1)

        return all_scores

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

def compute_ppl(texts, model, tokenizer, device):
    ppl_list = []
    for txt in tqdm(texts, desc="Calculating PPL"):
        enc = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss
        ppl = torch.exp(loss).item()
        ppl_list.append(ppl)
    return ppl_list

api = PerspectiveAPI()
toxicity_results = api.request(texts, batch_size=50)

with open("toxicity_results.json", "w", encoding="utf-8") as f:
    for r in toxicity_results:
        f.write(json.dumps(r))
        f.write("\n")

ppl_scores = compute_ppl(texts, model, tokenizer, device)

toxicity_avg = {k: sum(d["scores"][k] for d in toxicity_results)/len(toxicity_results) for k in ATTRIBUTES}
ppl_avg = sum(ppl_scores)/len(ppl_scores)

print(f"Average PPL: {ppl_avg:.2f}")
first_attr = list(toxicity_avg.keys())[0]
print(f"{first_attr}: {toxicity_avg[first_attr]:.4f}")