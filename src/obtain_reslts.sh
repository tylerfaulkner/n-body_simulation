#!/bin/bash

mkdir "timeResults"
steps=1024

for size in 256 512 1024 2048 4096 8192
do
    mkdir "./timeResults/${size}"
    cpu_file="./timeResults/${size}/cpu_results.csv"
    gpu_file="./timeResults/${size}/gpu_results.csv"
    end_file="./timeTesults/${size}_results.csv"
    rm $cpu_file
    rm $gpu_file
    rm $end_file
    for run in 1 2 3 4 5 6 7 8 9 10
    do
        echo Running Size:$size Trial:$run
        resultString=`./n-body_simulation $size $steps | grep ms | egrep -o '[0-9]+.[0-9]+'`
        #echo $resultString
        readarray -t results <<<$resultString
        echo ${results[0]} ${results[1]}
        if [[ $run -ne 1 ]]; then
            echo -n "," >> $cpu_file
            echo -n "," >> $gpu_file
        fi
        echo -n "${results[0]}" >> $cpu_file
        echo -n "${results[1]}" >> $gpu_file
    done
    cat $cpu_file >> $end_file
    echo -ne "\n" >> $end_file
    cat $gpu_file >> $end_file
done
