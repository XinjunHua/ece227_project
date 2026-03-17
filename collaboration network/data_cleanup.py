import csv

with open("CA-HepTh.txt") as fin, open("edges.csv", "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["Source", "Target"])
    for line in fin:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            writer.writerow(parts)