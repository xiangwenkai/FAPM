# prepare fasta data
name_list = ['P18281']
sequence_list = ['MNPELQSAIGQGAALKHAETVDKSAPQIENVTVKKVDRSSFLEEVAKPHELKHAETVDKSGPAIPEDVHVKKVDRGAFLSEIEKAAKQ']
with open('example.fasta', 'w') as f:
    for i, j in zip(name_list, sequence_list):
        f.write('>{}\n'.format(i))
        f.write('{}\n'.format(j))