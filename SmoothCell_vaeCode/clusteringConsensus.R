#####################################################################################################
##### clusteringConsensus.R
# Go over clutering results in directory ..._CLUSTERsearch_TF_VAE, and pick the number of clusters
# to work with as the most common number
##### updated by: Jose Malagon Lopez, 2018-09
#####################################################################################################
suppressMessages(library(optparse))
suppressMessages(library(data.table)); suppressMessages(library(dplyr)); suppressMessages(library(feather))
suppressMessages(library(igraph)); suppressMessages(library(SNFtool))
#####################################################################################################
#####################################################################################################
options(scipen=999)
#####################################################################################################
### Parameters
option_list = list(
  make_option(c("--output_dir"), type="character", help="full path to data folder.", metavar="character"),
  make_option(c("--logNormData"), type="character", help="main data file name.", metavar="character"),
  make_option(c("--output_name"), type="character", default='test', help="output name", metavar="character")
)
opt = parse_args(OptionParser(option_list=option_list))
#####################################################################################################
########################################### Main objects ############################################
# data allocation
output_dir = opt$output_dir
logNormData = opt$logNormData
output_name = opt$output_name
#####################################################################################################
############################################## SCRIPT ###############################################
#####################################################################################################
# clusteringConsensus: Loading Data.

data_folder =  file.path(output_dir, 'outputData')

if (!(dir.exists(data_folder))) {
  dir.create(data_folder)
}

# count data; columns=samples; rows=genes
scData.count = as.data.frame(read_feather(logNormData))

rm(logNormData)
##################################################################################################### 
#####################################################################################################
# clusteringConsensus: Consensus cluster: data location.

search_dir = list.files(path=output_dir, pattern = 'CLUSTERsearch_TF_VAE', recursive = FALSE, include.dirs = TRUE)

all_stats = read.delim(file.path(output_dir, search_dir, 'all_stats_list.txt'))
all_stats = all_stats[all_stats$Warning != 1,]
all_stats = droplevels(all_stats)

run_dirs_all = list.files(path=file.path(output_dir, search_dir), pattern = "model_run")
sort_idx = order(as.numeric(sapply(strsplit(run_dirs_all, split = '_'), '[', 3)), decreasing = FALSE)
run_dirs_all = run_dirs_all[sort_idx]

rm(sort_idx)
##################################################################################################### 
#####################################################################################################
# select adequate number of clusters

selecting_frame = data.frame(clusterCount = sort(unique(all_stats$Total_Clusters)))
selecting_frame$totalRuns = 0
selecting_frame$totalValidLL = 0
for (i in 1:nrow(selecting_frame))
{
  cl_count = selecting_frame[i,'clusterCount']
  selecting_frame[i,'totalRuns'] = nrow(all_stats[all_stats$Total_Clusters == cl_count,])
  selecting_frame[i,'totalValidLL'] = sum(all_stats[all_stats$Total_Clusters == cl_count,'Valid_LL'])
}
selecting_frame = selecting_frame[order(-selecting_frame$totalRuns, selecting_frame$totalValidLL),]
best_clust_num = selecting_frame$clusterCount[1]
run_dirs = run_dirs_all[which(all_stats$Total_Clusters == best_clust_num)]

rm(i, cl_count, selecting_frame)
##################################################################################################### 
#####################################################################################################
# cat("clusteringConsensus: Consensus cluster: Spectral CLustering. \n")

# adjency matrix
adj_mats = matrix(0L, nrow = nrow(scData.count), ncol = nrow(scData.count))

for (i in 1:length(run_dirs))
{
  cluster_labels = fread(file.path(output_dir, search_dir, run_dirs[i], 'vae_model', 'phenograph_cluster.txt'),
                         sep = '\t', header = FALSE, data.table = FALSE)
  cluster_labels = cluster_labels$V1
  n_clusters = unique(cluster_labels)
  
  combos_list = list()
  for (j in 1:length(n_clusters))
  {
    clust_idx = which(cluster_labels == n_clusters[j])
    combos_list[[j]] = as.data.frame(t(combn(clust_idx,2)))
  }

  all_combos = bind_rows(combos_list)
  eg = graph_from_edgelist(as.matrix(all_combos), directed = FALSE)
  adj_mat = as_adjacency_matrix(eg, type='both', sparse = FALSE)
  diag(adj_mat) = 1
  adj_mats = adj_mats + adj_mat
}

mean_adj = adj_mats / length(run_dirs)

# spectral clustering
specc_clust = spectralClustering(mean_adj, best_clust_num)
specc_clust = specc_clust - 1

rm(cluster_labels, n_clusters, all_combos, eg, adj_mat, combos_list, i, j)
##################################################################################################### 
#####################################################################################################
# cat("clusteringConsensus: Consensus cluster: saving data. \n")

write.table(specc_clust, file.path(data_folder, paste(output_name,'filtered_best_clust',best_clust_num,'spectral_consensus_cluster.txt',sep='.')),
            row.names = FALSE, col.names = FALSE, quote = FALSE, sep='\t')
#####################################################################################################
#####################################################################################################
# cat("clusteringConsensus: done!. \n")
#####################################################################################################
#####################################################################################################
