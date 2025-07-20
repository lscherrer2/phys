#!/bin/bash
for file in $(ls ./test/);
do python ./test/$file;
done;
echo "complete"
