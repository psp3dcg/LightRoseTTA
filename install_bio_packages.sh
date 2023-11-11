#!/bin/bash

# download hh-suite
echo "downloading hhsuite..."
wget https://github.com/soedinglab/hh-suite/releases/download/v3.2.0/hhsuite-3.2.0-AVX2-Linux.tar.gz --no-check-certificate
tar -xzvf hhsuite-3.2.0-AVX2-Linux.tar.gz
mkdir hhsuite
mv bin hhsuite
mv data hhsuite
mv scripts hhsuite
mv LICENSE hhsuite
mv README hhsuite
export PATH="$(pwd)/hhsuite/bin:$(pwd)/hhsuite/scripts:$PATH"
echo 'PATH=$(pwd)/hhsuite/bin:$(pwd)/hhsuite/scripts:$PATH'>>/etc/profile
# Backer version

# download blast-2.2.26
echo "downloading blast-2.2.26..."
wget ftp://ftp.ncbi.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz
mkdir -p blast-2.2.26
tar xvf blast-2.2.26 -C blast-2.2.26
export BLAST_HOME=$(pwd)/blast-2.2.26
# cp -r blast-2.2.26 your_path/preprocess_code/msa_feat

# download psipred
echo "downloading psipred"
git clone https://github.com/psipred/psipred.git
export PSIBLAST=$(pwd)/psipred-master
export PATH=$BLAST_HOME/bin:$PSIBLAST/bin:$PATH
# cp -r psipred your_path/preprocess_code/msa_feat


# download cs-blast
case "$(uname -s)" in
    Linux*)     platform=linux;;
    Darwin*)    platform=macosx;;
    *)          echo "unsupported OS type. exiting"; exit 1
esac
echo "installing for ${platform}"

# the cs-blast platform descriptoin includes the width of memory addresses
# we expect a 64-bit operating system
if [[ ${platform} == "linux" ]]; then
    platform=${platform}64
fi

echo "downloading cs-blast . . ."
wget http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_${platform}.tar.gz -O csblast-2.2.3.tar.gz
mkdir -p csblast-2.2.3
tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1
# cp -r csblast-2.2.3 your_path/preprocess_code/msa_feat