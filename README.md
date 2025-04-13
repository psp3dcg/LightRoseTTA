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


## Test datasets

Download the [test datasets](https://drive.google.com/drive/folders/1n_RgI_OpyPHOEQw7P8K9H01f5guVpxhv?usp=sharing) from google drive



## Testing steps

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
python test_script.py ./Orphan25_fasta ./Orphan25_data ./Uniref30_2020_06 ./pdb100_2021Mar03 ./BFD ./weights/LightRoseTTA.pth

# for antibodies
python test_script.py ./Antibody_fasta ./Antibody_data ./Uniref30_2020_06 ./pdb100_2021Mar03 ./BFD ./weights/LightRoseTTA-Ab.pth

The output "*.pdb" files are located in "data_write_path/predict_pdb" (e.g. Orphan25_data/predict_pdb)
```

## Train dataset
Download the [[train dataset](https://drive.google.com/file/d/1HYIZHXyB30adqkKUO7Fe5m4bT_6nFvTX/view?usp=drive_link)] from google drive


## Training steps
```
# using LightRoseTTA train dataset
(a) download the LightRoseTTA_train_data.tar.gz from the given google drive link.
(b) uncompress the download file.

# build your own train dataset
(a) download the LightRoseTTA_preprocess_train_data.zip and unzip it
(b) prepare the ".fasta" files and corresponding ".pdb" files
(c) cd "LightRoseTTA_preprocess_train_data" folder and generate data following the README.md

# run the training python file
(a) download the LightRoseTTA_training_code.zip and unzip it
(b) cd "LightRoseTTA_training_code" folder
(c) python LightRoseTTA_train.py -dataset [training_data_path]
ps: training_data_path should include "raw" folder and "processed" folder
```

## References
```
@article{wang2025lightrosetta,
  title={LightRoseTTA: High-Efficient and Accurate Protein Structure Prediction Using a Light-Weight Deep Graph Model},
  author={Wang, Xudong and Zhang, Tong and Liu, Guangbu and Cui, Zhen and Zeng, Zhiyong and Long, Cheng and Zheng, Wenming and Yang, Jian},
  journal={Advanced Science},
  pages={2309051},
  year={2025},
  publisher={Wiley Online Library}
}
```




	
