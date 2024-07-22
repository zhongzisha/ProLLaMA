#!/bin/bash

#SBATCH --job-name=debug2
#SBATCh --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=norm
#SBATCH --gres=lscratch:600
##SBATCH --gres=gpu:a100:2,lscratch:600
#SBATCH --time=200:00:00
##SBATCH --exclusive
##SBATCH --output=%x-%j.out
##SBATCH --export=ALL


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
    if [ "1" == "1" ]; then
        source /data/zhongz2/anaconda3/bin/activate th21_ds0144
        module load CUDA/12.1
        module load cuDNN/8.9.2/CUDA-12
        module load gcc/11.3.0
        CODE_ROOT=/home/$USER/LLaVA
        export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
    fi
fi


if [ "0" == "1" ]; then  # generate data
cd $MYTMP_DIR
cp /data/zhongz2/temp29/uniref50.fasta.gz .
gzip --decompress uniref50.fasta.gz

cd /home/zhongz2/ProLLaMA
python create_pretrain_dataset.py $MYTMP_DIR/uniref50.fasta $MYTMP_DIR/uniref50_txts
wait;
python generate_cache.py $MYTMP_DIR/uniref50_txts
fi

exit;

