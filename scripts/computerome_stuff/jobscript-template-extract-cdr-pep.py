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
#PBS -l nodes=1:ppn=14
###
### Required RAM in GB
#PBS -l mem=50GB
###
### How long (max) will the job take, here 24 hours
#PBS -l walltime=1:00:00
###
### Output files - not required to be specified
### Comment out the next 2 lines to use the job id instead in the file names
#PBS -e /home/projects/ht3_aim/people/sebdel/masters/scripts/computerome_stuff/extract_cdr_pep.log
#PBS -o /home/projects/ht3_aim/people/sebdel/masters/scripts/computerome_stuff/extract_cdr_pep_err.log
###
### Job name - not required to be specified
### It is often easier just to use the job id instead for recognition
#PBS -N extract_cdr_pep
###
### More qsub options can be added here


# This part is the real job script
# Here follows the user commands:

# Load all required modules for the job
module load tools 
module load miniconda3/4.10.3
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/projects/ht3_aim/people/sebdel/envs/envs/proteinsolver/

#python3 /home/projects/ht3_aim/people/sebdel/masters/scripts/generate_data/extract_cdr_pep_from_embedding.py -s ps
#python3 /home/projects/ht3_aim/people/sebdel/masters/scripts/generate_data/extract_cdr_pep_from_embedding.py -s esm
#python3 /home/projects/ht3_aim/people/sebdel/masters/scripts/generate_data/extract_cdr_pep_from_embedding.py -s esm_ps
#python3 /home/projects/ht3_aim/people/sebdel/masters/scripts/generate_data/extract_cdr_pep_from_embedding.py -s blosum
python3 /home/projects/ht3_aim/people/sebdel/masters/scripts/generate_data/extract_cdr_pep_energy.py
