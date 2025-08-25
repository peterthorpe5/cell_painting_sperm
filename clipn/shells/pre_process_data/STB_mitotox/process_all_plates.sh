#!/bin/env bash

##$ -adds l_hard gpu 1
##$ -adds l_hard cuda.0.name 'NVIDIA A40'
#$ -j y
#$ -N process_all_plates
#$ -l h_vmem=200G
#$ -cwd


cd /home/pthorpe/scratch/laura


