#!/bin/sh
### Every line in this header section should start with ### for a comment
### or #PBS for an option for qsub
### Note: No unix commands may be executed until after the last #PBS line
###
### Account information
#PBS -W group_list=ht3_aim -A ht3_aim
##
### Send mail when job is aborted or terminates abnormally
#PBS -M s163691@student.dtu.dk
#PBS -m abe
###
### Compute resources, here 1 core on 1 node
#PBS -l nodes=1:ppn=4
###
### Required RAM in GB
#PBS -l mem=50GB
###
### How long (max) will the job take, here 24 hours
#PBS -l walltime=150:00:00
###
### Output files - not required to be specified
### Comment out the next 2 lines to use the job id instead in the file names
#PBS -e /home/projects/ht3_aim/people/sebdel/masters/scripts/computerome_stuff/cluster_seq_err.log
#PBS -o /home/projects/ht3_aim/people/sebdel/masters/scripts/computerome_stuff/cluster_seq.log
###
### Job name - not required to be specified
### It is often easier just to use the job id instead for recognition
#PBS -N cluster_seq
###
### More qsub options can be added here


# This part is the real job script
# Here follows the user commands:

# Load all required modules for the job
module load tools 
module load miniconda3/4.10.3
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/projects/ht3_aim/people/sebdel/envs/envs/proteinsolver/
mmseqs easy-cluster /home/people/sebdel/ht3_aim/masters/data/neat_data/cdr3b_seqs.fsa /home/people/sebdel/ht3_aim/masters/data/neat_data/clusterRes_cdr3b_raw_idx tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1
mmseqs easy-cluster /home/people/sebdel/ht3_aim/masters/data/neat_data/cdr3b_seqs.fsa /home/people/sebdel/ht3_aim/masters/data/neat_data/clusterRes_cdr3b_raw_idx_low_cov tmp --min-seq-id 0.5 -c 0.5 --cov-mode 1

#mmseqs easy-cluster /home/people/sebdel/ht3_aim/masters/data/neat_data/cdr3b_seqs.fsa /home/people/sebdel/ht3_aim/masters/data/neat_data/clusterRes_cdr3b_test_cov_95 tmp --min-seq-id 0.8 -c 0.95 --cov-mode 1

#mmseqs easy-cluster /home/people/sebdel/ht3_aim/masters/data/neat_data/cdr3b_seqs.fsa /home/people/sebdel/ht3_aim/masters/data/neat_data/clusterRes_cdr3b_test_cov_25 tmp --min-seq-id 0.8 -c 0.25 --cov-mode 1

#mmseqs easy-cluster /home/people/sebdel/ht3_aim/masters/data/neat_data/cdr3b_seqs.fsa /home/people/sebdel/ht3_aim/masters/data/neat_data/clusterRes_cdr3b_test_cov_95_mode_2 tmp --min-seq-id 0.8 -c 0.95 --cov-mode 2

#mmseqs easy-cluster /home/people/sebdel/ht3_aim/masters/data/neat_data/cdr3b_seqs.fsa /home/people/sebdel/ht3_aim/masters/data/neat_data/clusterRes_cdr3b_test_cov_25_mode_2 tmp --min-seq-id 0.8 -c 0.25 --cov-mode 2
