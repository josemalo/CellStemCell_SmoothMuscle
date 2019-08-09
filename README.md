# CellStemCell_SmoothMuscle
Code used for the analysis of the scRNA-seq data for the article "Smooth muscle cell reprogramming in aortic aneurysms", Cell Stem Cell, 2019.

## Variational Auto-Encoder<a name="VAE"></a>

### Dependencies<a href="VAEdependencies"></a>

* Python (3.5.2)
* R (3.2.3)

### Usage<a name="VAEusage"></a>
The following command line
```
python3 /path/to/variationalAutoEncoder.py --act_fnc relu --batch 256 --cuda_device 0 --early_stop 50 --epochs 5000 --ll_samples 100 --min_epoch 250 --name "name" --nn_dims 32 32 32 --output_dir "output_dir" --post_layer zinb --reg mmd --test_fraction 0.2 --warm_up 0.01 --n_reps 100 --hp_samples 100 --hp_layers 2 4 --hp_dims 512 256 128 64 32 --hp_lr -4.5 -3 --hp_dropout 0.6 0.9 --hp_latent 2 24 --code_folder "code_folder" --data_input "data_input"
```
will run the entire pipeline, where

- `name`: Run's name. It will be appended to the output files. 
- `output_dir`: The absolute path to the folder in which results will be stored.
- `code_folder`: path/to/NatureMetabolism_Endothelial/Endothelial_vaeCode.
- `data_input`: path/to/csv file with TMM data to be analyzed. Provided in this repositoty.

### Output<a name="VAEoutput_folder"></a>

There will be three output folders. The following is a brief description of the more important files:

1. `outputData`: 
(a) *name.feather*, the formatted input data as a feather file; 
(b) a txt file with the clustering labels.

2. `DATE_name_HPsearch_TF_VAE`: 
(a) subfolders with the results, one for every run; 
(b) "all_stats_list.txt": txt file with all the statistic concerning the performance of the run; 
(c) "hyperparameters.txt": txt file with hyper-parameters used; 
(d) "FailedHyperparameters.txt": txt file with the hyper-parameters inducing non-convergent runs (if any). 

3. `DATE_name_CLUSTERsearch_TF_VAE`: 
(a) subfolders with the results, one for every run; 
(b) "all_stats_list.txt": txt file with all the statistic concerning the performance of the run.


#### Folder With Results<a name="VAEresult_folder"></a>

The most important results for any run are the following:

- `run_N/all_data_idx.txt`: File with the ids of the cells used as the traning data. 

- `run_N/test_data_idx.txt`: File with the ids of the cells used as the testing data. 

- `run_N/vae_model/phenograph_cluster.txt`: File with the cluster labels associated to every cell at the given run.

- `run_N/vae_model/x_latent_final.feather`: File with the latent space at the given run. 


## *scdiff* Adaptation<a name="SCDIFF"></a>

### Dependencies<a href="SCDIFFdependencies"></a>

* Python (2.7.15)

### Usage<a name="SCDIFFusage"></a>
The following command line
```
python /path/to/scdiffAltRun.py --mainInput "mainInput" --clusterInfo "clusterInfo" --clusterID "clusterID" --TF "TF" --outName "outName" --outputDir "outputDir"
```
will run the *scdiff* pipeline, where

- `mainInput`: path/to/SMC1_treatment_scdiff_MainInput.txt, input single cell RNA-seq expression data as required for 
*scdiff*. 
This is a file with header where cell ID, cell Time and cell Label are the first three columns, respectively, and the 
remaining columns are the gene expression values. Provided in this repositoty.
- `clusterInfo`: path/to/SMC1_treatment_scdiff_ClusterInput.txt, cluster configuration as required for *scdiff*, which 
specifies the custom initial clustering parameters. 
This i sa file without header which columns are Time and Number_of_Cluster. Provided in this repositoty.
- `clusterID`: path/to/SMC1_treatment_scdiff_vaeLabels.txt, cluster labels determined by the variational auto-encoder. 
Provided in this repositoty.
- `TF`: path/to/Mouse_TF_targets.txt, TF-DNA interactions used in the analysis and taken from the *scdiff* repository.
Provided in this repositoty.
- `outName`: Run's name. It will be appended to the output files. 
- `outputDir`: The absolute path to the folder in which results will be stored.

### Output<a name="SCDIFFoutput_folders"></a>

There will be three output files:

(a) `outName_Nodes.txt`. File with the information regarding to the cell clusters identified by the variational auto-encoder
under the differentiation network analysis. 

(b) `outName_Cells.txt`. File with the information regarding to the cells under the differentiation network analysis. 

(c) `outName_Paths.txt`. File with the information regarding to the edges in the differentiation graph. 
