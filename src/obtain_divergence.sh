#!/bin/bash
folder="divResults"
mkdir "${folder}"
steps=1024

for size in 256 512 1024 2048 4196 8192
do
    end_file="./${folder}/${size}_results.csv"
    rm $end_file
    for run in 1 2 3 4 5
    do
        echo Running Size:$size Trial:$run
        resultString=`./n-body_simulation $size $steps | grep units | egrep -o '[0-9]+.[0-9]+'`
        #echo $resultString
        readarray -t results <<<$resultString
        echo ${results[0]}
        if [[ $run -ne 1 ]]; then
            echo -n "," >> $end_file
        fi
        echo -n "${results[0]}" >> $end_file
    done
done
