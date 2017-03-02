#!/bin/bash

[ "$USER" == "pnef" ]     && WorkDir=/u/at/pnef/Work/Code/Reclustering/
[ "$USER" == "swiatlow" ] && WorkDir=/u/at/swiatlow/nfs/projects/Reclustering/
[ "$USER" == "bpn7" ]     && WorkDir=/nfs/slac/g/atlas/u01/users/bnachman/SLAC_pythia/Reclustering/
# add similar line if you are not pnef

SubFileLoc=`pwd`/_batchSingleSub.sh
#rm $SubFileLoc
DateSuffix=`date +%Y%m%d_%Hh%Mmin`

echo '#!/bin/bash
echo CD to $1
echo CMD is $2

cd $1
source setup.sh
cmd=$4

echo MAKING TEMP DIR $2
JOBFILEDIR=$2
mkdir $JOBFILEDIR
REALOUT=$3
echo MADE TEMP DIR $JOBFILEDIR
echo WILL COPY TO $REALOUT

shift
shift
echo Calling $cmd $*
$cmd $*
cp -r $JOBFILEDIR/*.root $REALOUT
echo COPYING to $REALOUT
rm -rf $JOBFILEDIR
' > $SubFileLoc
chmod u+x $SubFileLoc

#----------------
Process=2
pThatMin=200
pThatMax=400
BosonMass=800
for mu in 40; do
    Queue=long
    nevents=1000
    njobs=100
    [ "$mu" == "40" ]  && Queue=xlong  && njobs=200  && nevents=500
    [ "$mu" == "60" ]  && Queue=xlong  && njobs=200  && nevents=500
    [ "$mu" == "80" ]  && Queue=xlong  && njobs=200  && nevents=500
    LogPrefix=`pwd`/logs/${DateSuffix}/${DateSuffix}_bsub_${mu}_
    OutDirFinal=`pwd`/files/${DateSuffix}
    mkdir -p `dirname $LogPrefix`
    mkdir -p $OutDirFinal
    echo
    echo "Submitting $njobs jobs each with $nevents events to $Queue"
    echo $LogPrefix
    for (( ii=1; ii<=$njobs; ii++ )) ;  do
        echo $ii
        OutDir=/scratch/${DateSuffix}_${ii}/
        bsub -q ${Queue} -R 'select[(!preempt&&rhel60&&cvmfs&&inet)]' -o $LogPrefix${ii}.log $SubFileLoc           \
            ${WorkDir} ${OutDir} ${OutDirFinal} ./Recluster.exe  \
            --Pileup $mu                 \
            --OutFile ${OutDir}/Sample_mu_${mu}_nevents_${nevents}_job_${ii}_Process_${Process}_${pThatMin}_${pThatMax}.root \
            --Proc ${Process} \
            --NEvents ${nevents} \
            --pThatMin ${pThatMin} \
            --pThatMax ${pThatMax} \
            --BosonMass ${BosonMass} 

    
    done
done

