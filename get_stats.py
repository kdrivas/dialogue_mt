import os
import json

for corpus in ["wmtchat2020"]:
    for partition in ["train", "dev", "test"]:
        with open(f"data/preprocessed/{n_file}/{partition}.json") as f:
            data = json.load(f)

        cont_conv = 0
        cont_utter = 0
        for conversation in data:
            cont_conv += 1
            for utterance in conversation:
                cont_utter += 1

        print(n_file, partition, cont_conv, cont_utter)
