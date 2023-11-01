#!/bin/bash
echo "Starting shell script..."
## 1.1.1
firstline=$(head -n 1 source/changelog.md)
read -a splitfirstline <<< $firstline
version=${splitfirstline[1]}
echo "You are building on version" $version
echo "Continue? Enter 1 for yes, 0 for no"
read versioncontinue
if [ $versioncontinue -eq 1 ]
then
  echo "OK"
  for filename in source/*
  do
    if [ "$filename" == "source/secretinfo.md" ]
    then
      echo "Not copying" $filename
    else 
      echo "Copying..." $filename
      cp $filename build/
    fi
  done
  cd build/
  echo "This build version contains:"
  ls -l
else
  echo "Please come back when you are ready."
fi