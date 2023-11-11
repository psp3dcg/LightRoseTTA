import os
import sys



def delete_struct_in_hhr(file_path, write_path, protein_name):
        protein_name = protein_name.split('_')[0].lower()
        with open(file_path) as f:
                hhr_file = f.readlines()

        new_hhr_file = []
        delete_protein_number = []
        del_flag = False
        for line in hhr_file:
                if line[4:8] == protein_name:
                        delete_protein_number.append(int(line[:3]))
                        continue

                if line[:3] == 'No ':
                        if int(line[3:]) in delete_protein_number:
                                del_flag = True
                        else:
                                del_flag = False
                if del_flag:
                        continue



                new_hhr_file.append(line)

        with open(write_path, 'w') as f:
                for line in new_hhr_file:
                        f.write(line)

def delete_struct_in_atab(file_path, write_path, protein_name):
        protein_name = protein_name.split('_')[0].lower()
        with open(file_path) as f:
                atab_file = f.readlines()

        new_atab_file = []
        del_flag = False
        for line in atab_file:
                if line[:1] == '>':
                        del_flag = False

                if line[1:5] == protein_name:
                        del_flag = True
                        continue

                if del_flag:
                        continue

                new_atab_file.append(line)

        with open(write_path, 'w') as f:
                for line in new_atab_file:
                        f.write(line)




