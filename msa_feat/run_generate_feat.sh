#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e
#export PATH="$(pwd)/hhsuite/bin:$(pwd)/hhsuite/scripts:$PATH"
#export BLAST_HOME=$(pwd)/blast-2.2.26
#export PSIBLAST=$(pwd)/psipred-master
#export PATH=$BLAST_HOME/bin:$PSIBLAST/bin:$PATH
############################################################
SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`

CPU="8"  # number of CPUs to use
MEM="64" # max memory (in GB)

# Inputs:
IN="$1"                # input.fasta
WDIR=`realpath -s $2`  # working folder
s_DB="$3"             # sequence search database
t_DB="$4"             # template search database
bs_DB="$5"            # big sequence search database


LEN=`tail -n1 $IN | wc -m`

mkdir -p $WDIR/log

############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits"
    bash $PIPEDIR/make_msa.sh $IN $WDIR $CPU $MEM $s_DB $bs_DB 
    #> $WDIR/log/make_msa.stdout 2> $WDIR/log/make_msa.stderr
fi

###########################################################
# 2. predict secondary structure for HHsearch run
############################################################
if [ ! -s $WDIR/t000_.ss2 ]
then
    echo "Running PSIPRED"
    bash $PIPEDIR/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 
    #> $WDIR/log/make_ss.stdout 2> $WDIR/log/make_ss.stderr
fi


############################################################
# 3. search for templates
############################################################
# DB="data3/protein/datasets-search/pdb100_2021Mar03/pdb100_2021Mar03"
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch"
    HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $t_DB"
    cat $WDIR/t000_.ss2 $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
    $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -atab $WDIR/t000_.atab -v 0 
    #> $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
fi




