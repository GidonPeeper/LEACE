with open("/home/gpeeper/LEACE/data/narratives/test.conllu") as f:
    for i, line in enumerate(f, 1):
        if line.strip() and not line.startswith("#"):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10:
                print(f"Line {i} has {len(fields)} fields: {fields}")
