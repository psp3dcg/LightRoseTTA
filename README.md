## LightRoseTTA - Pytorch

## Installation

1. Clone the package
```
git clone https://github.com/psp3dcg/LightRoseTTA.git
cd LightRoseTTA
```

2. Create conda environment using `LightRoseTTA-env.yml` file
```
# create conda environment for LightRoseTTA
conda env create -f LightRoseTTA-env.yml
```

3. Download and Install the biological packages

```bash
$ bash install_bio_package.sh
$ copy bio_package(blast, csblast, psipred) to your_path/LightRoseTTA/msa_feat

```

4. Download the Uniref30[46G], BFD[272G] and pdb100[over 100G] datasets

```bash
$ python download_datasets.py

```

## Usage

```
# run the testing python file
python test_script.py [FASTA_folder_path] [data_write_path] [Uniref30_dataset_path]  [pdb100_dataset_path] [BFD_dataset_path] [model_file_path]
	
-FASTA_folder_path: the path of folder containing FASTA files
-data_write_path: the path of folder to write generated data
-Uniref30_dataset_path: the path of Uniref30 dataset
-pdb100_dataset_path: the path of pdb100 dataset
-BFD_dataset_path: the path of BFD dataset
-model_file_path: the path of model file

For example,
# for general proteins
python test_LightRoseTTA_from_fasta.py ./Orphan25_fasta ./Orphan25_data ./Uniref30_2020_06 ./pdb100_2021Mar03 ./BFD ./weights/LightRoseTTA.pth

# for antibodies
python test_LightRoseTTA_from_fasta.py ./Antibody_fasta ./Antibody_data ./Uniref30_2020_06 ./pdb100_2021Mar03 ./BFD ./weights/LightRoseTTA-Ab.pth

The output "*.pdb" files are located in "data_write_path/predict_pdb" (e.g. Orphan25_data/predict_pdb)
```







	
