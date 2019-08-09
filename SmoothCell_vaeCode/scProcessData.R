#####################################################################################################
##### scProcessData.R
# Format TMM matrix (rows=samples, columns=genes)
#####
#####################################################################################################
suppressMessages(library(optparse))
suppressMessages(library(data.table)); suppressMessages(library(dplyr)); suppressMessages(library(feather))
#####################################################################################################
#####################################################################################################
options(scipen=999)
#####################################################################################################
### Parameters
option_list = list(
  make_option(c("--output_dir"), type="character", help="full path to data folder.", metavar="character"),
  make_option(c("--data_input"), type="character", help="full path to data file name.", metavar="character"),
  make_option(c("--output_name"), type="character", default='test', help="output name", metavar="character")
)
opt = parse_args(OptionParser(option_list=option_list))
#####################################################################################################
########################################### Main objects ############################################ 
# data allocation 
output_dir = opt$output_dir
data_input = opt$data_input
output_name = opt$output_name
#####################################################################################################
############################################## SCRIPT ############################################### 
#####################################################################################################

data_folder = file.path(output_dir, 'outputData')

# count data; columns=samples; rows=genes
scData.full = read.csv(data_input, sep = ',')
scData.count = scData.full[,2:ncol(scData.full)]
colnames(scData.count) = gsub('\\.', '_', colnames(scData.count))

rm(scData.full, data_input)
############################################## SCRIPT ###############################################
#####################################################################################################
# write("scProcessData: Saving Data. \n", stderr())

write_feather(scData.count, file.path(data_folder, paste(output_name, 'feather', sep='.')))
#####################################################################################################
#####################################################################################################
# write("scProcessData: done!. \n", stderr())
#####################################################################################################
#####################################################################################################
