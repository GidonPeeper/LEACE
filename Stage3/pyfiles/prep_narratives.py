import os
import glob
import spacy

# Load spaCy model with POS and dependency parsing
nlp = spacy.load("en_core_web_sm")

# Input/output paths
txt_dir = "/Users/gidonp/transcripts"
txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
output_conllu = os.path.join(txt_dir, "narratives.conllu")

# Concatenate all text files
all_text = ""
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        all_text += f.read() + "\n"

# Process the full text
doc = nlp(all_text)

# Write CoNLL-U
with open(output_conllu, "w", encoding="utf-8") as out_f:
    for sent_id, sent in enumerate(doc.sents, start=1):
        out_f.write(f"# sent_id = {sent_id}\n")
        out_f.write(f"# text = {sent.text}\n")
        for i, token in enumerate(sent):
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
            out_f.write('\t'.join(fields) + '\n')
        out_f.write('\n')  # sentence separator

print(f"Finished writing CoNLL-U to: {output_conllu}")
