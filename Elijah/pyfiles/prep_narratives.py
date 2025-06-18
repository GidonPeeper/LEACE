import os
import glob
import spacy
import random

# Load spaCy model with POS and dependency parsing
nlp = spacy.load("en_core_web_sm")

# Input/output paths
txt_dir = "/home/gpeeper/LEACE/data/narratives/transcripts"
txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
output_dir = "/home/gpeeper/LEACE/data/narratives"
os.makedirs(output_dir, exist_ok=True)

# Concatenate all text files
all_text = ""
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        all_text += f.read() + "\n"

print("Found txt files:", txt_files)
print("Total characters in all_text:", len(all_text))

# Process the full text
doc = nlp(all_text)
sentences = list(doc.sents)
num_sentences = len(sentences)
indices = list(range(num_sentences))
random.seed(42)
random.shuffle(indices)

print("Number of sentences found:", num_sentences)

# 80/20 split
split_idx = int(0.8 * num_sentences)
train_indices = set(indices[:split_idx])
test_indices = set(indices[split_idx:])

print("Train indices:", len(train_indices))
print("Test indices:", len(test_indices))

def write_conllu(sent_indices, out_path):
    with open(out_path, "w", encoding="utf-8") as out_f:
        for sent_id, idx in enumerate(sent_indices, start=1):
            sent = sentences[idx]
            # Skip sentences with no tokens or only one token that is the whole sentence
            valid_tokens = [token for token in sent if not token.is_space and token.text.strip()]
            if len(valid_tokens) == 0:
                continue
            # If the only token is the whole sentence, skip
            if len(valid_tokens) == 1 and valid_tokens[0].text.strip() == sent.text.strip():
                continue
            out_f.write(f"# sent_id = {sent_id}\n")
            out_f.write(f"# text = {sent.text}\n")
            for i, token in enumerate(sent):
                if token.is_space or not token.text.strip():
                    continue
                ID = i + 1
                FORM = token.text
                LEMMA = token.lemma_
                UPOS = token.pos_
                XPOS = token.tag_
                FEATS = str(token.morph) if token.morph else "_"
                HEAD = 0 if token.head == token else token.head.i - sent.start + 1
                DEPREL = token.dep_
                DEPS = "_"
                MISC = "_"
                fields = [str(ID), FORM, LEMMA, UPOS, XPOS, FEATS, str(HEAD), DEPREL, DEPS, MISC]
                if all(f != "" for f in fields) and len(fields) == 10:
                    out_f.write('\t'.join(fields) + '\n')
            out_f.write('\n')  # sentence separator

# Write train and test splits
train_path = os.path.join(output_dir, "train.conllu")
test_path = os.path.join(output_dir, "test.conllu")
write_conllu(sorted(train_indices), train_path)
write_conllu(sorted(test_indices), test_path)

print(f"Finished writing train ({train_path}) and test ({test_path}) CoNLL-U files.")
