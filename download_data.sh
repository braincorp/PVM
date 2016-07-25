cd
echo "Downloading the labeled movie sequences (3GB). Be patient."
wget http://pvm.braincorporation.net/PVM_data_sequences.zip
unzip PVM_data_sequences.zip
rm PVM_data_sequences.zip
echo "Would you like to download the first set of pre-trained models? (1.6GB)"
read -r -p "Are you sure? [Y/n]" response
response=${response,,}
if [[ $response =~ ^(yes|y| ) ]]; then
    wget http://pvm.braincorporation.net/PVM_data_models01.zip
    unzip PVM_data_models01.zip
    rm PVM_data_models01.zip
fi
echo "Would you like to download the second set of pre-trained models? (2.7GB)"
read -r -p "Are you sure? [Y/n]" response
response=${response,,}
if [[ $response =~ ^(yes|y| ) ]]; then
    wget http://pvm.braincorporation.net/PVM_data_models02.zip
    unzip PVM_data_models02.zip
    rm PVM_data_models02.zip
fi
echo "Would you like to download the third set of pre-trained models? (1.8GB)"
read -r -p "Are you sure? [Y/n]" response
response=${response,,}
if [[ $response =~ ^(yes|y| ) ]]; then
    wget http://pvm.braincorporation.net/PVM_data_models03.zip
    unzip PVM_data_models03.zip
    rm PVM_data_models03.zip
fi
cd -