def remove_bad_sentences(conllu_path, bad_line_numbers, output_path):
    bad_line_numbers = set(bad_line_numbers)
    with open(conllu_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        current_sentence = []
        current_first_line = None
        current_line_number = 1
        for line in fin:
            if line.startswith("# sent_id"):
                # New sentence starts, check if previous should be written
                if current_sentence:
                    # If any line in the sentence is in bad_line_numbers, skip the sentence
                    if not any(lineno in bad_line_numbers for lineno, _ in current_sentence):
                        for _, l in current_sentence:
                            fout.write(l)
                    current_sentence = []
                    current_first_line = current_line_number
            current_sentence.append((current_line_number, line))
            if line.strip() == "":
                # End of sentence, check if it should be written
                if current_sentence:
                    if not any(lineno in bad_line_numbers for lineno, _ in current_sentence):
                        for _, l in current_sentence:
                            fout.write(l)
                    current_sentence = []
            current_line_number += 1
        # Write last sentence if needed
        if current_sentence:
            if not any(lineno in bad_line_numbers for lineno, _ in current_sentence):
                for _, l in current_sentence:
                    fout.write(l)

# --- Specify bad lines for each file ---
bad_train_lines = [
    11692, 12230, 12518, 12557, 12603, 12879, 12968, 12970, 13275, 13352, 23341,
    40895, 40956, 41204, 41324, 41435, 41437, 41773, 41841, 49491, 49529, 49575,
    49786, 49900, 49991, 50286, 50350, 51117
]
bad_test_lines = [
    3725, 10353, 12168
]

remove_bad_sentences(
    "/home/gpeeper/LEACE/data/narratives/train.conllu",
    bad_train_lines,
    "/home/gpeeper/LEACE/data/narratives/train_clean.conllu"
)
remove_bad_sentences(
    "/home/gpeeper/LEACE/data/narratives/test.conllu",
    bad_test_lines,
    "/home/gpeeper/LEACE/data/narratives/test_clean.conllu"
)

print("Done. Cleaned files written as train_clean.conllu and test_clean.conllu.")