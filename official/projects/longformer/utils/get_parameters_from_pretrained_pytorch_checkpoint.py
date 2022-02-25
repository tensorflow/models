import transformers
pretrained_lm = "allenai/longformer-base-4096"

model = transformers.AutoModel.from_pretrained(pretrained_lm)

import pickle
pickle.dump({
    n: p.data.numpy()
for n, p in model.named_parameters()}, open(f"{pretrained_lm.replace('/', '_')}.pk", "wb"))