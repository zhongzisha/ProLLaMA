import sys,os,gc
from Bio import SeqIO


input_fasta_file = sys.argv[1]  # 'path/to/uniref50.fasta'

# output_file = 'path/to/output_sequences.txt'
save_dir = sys.argv[2]
os.makedirs(save_dir, exist_ok=True)


# Set of 20 standard amino acids
standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# Function to check if a sequence contains only standard amino acids
def is_standard_amino_acid_sequence(sequence):
    return all(aa in standard_amino_acids for aa in sequence)

lines = []
c = 0
for record in SeqIO.parse(input_fasta_file, "fasta"):
    sequence = str(record.seq)
    if is_standard_amino_acid_sequence(sequence):
        lines.append('Seq=<{}>\n'.format(sequence))

    if len(lines) == 20000:
        with open(os.path.join(save_dir, 'input{}.txt'.format(c)), 'w') as fp:
            fp.writelines(lines)
        lines = []
        gc.collect()
        c += 1
        print(c)






