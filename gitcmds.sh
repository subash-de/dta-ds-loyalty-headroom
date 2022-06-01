#!/bin/bash
# Run this shell script to setup git and optionally perform an initial commit
echo "Please enter the repository url e.g. (https://dta-devops@dev.azure.com.../git@ssh.dev.azure.com...)"
echo "SSH is recommended but requires SSH keys to be setup"
read repo_url
echo "Initialising git"
git init
git remote add origin $repo_url
echo "Would you like to perform the initial commit?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) git add .; git commit -m 'initial commit'; git checkout -b develop; git push -u origin --all; echo "Setup completed"; break;;
        No ) echo "Setup completed"; exit;;
    esac
done