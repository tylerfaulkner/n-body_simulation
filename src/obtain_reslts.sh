#!/bin/bash

mkdir "timeResults"
steps=1024

for size in 256 512 1024 2048 4096 8192
do
    mkdir "./timeResults/${size}"
    cpu_file="./timeResults/${size}/cpu_results.csv"
    nt_file="./timeResults/${size}/tiled_results.csv"
    gpu_file="./timeResults/${size}/gpu_results.csv"
    end_file="./timeResults/${size}_results.csv"
    rm $nt_file
    rm $cpu_file
    rm $gpu_file
    rm $end_file
    for run in 1 2 3 4 5 6 7 8 9 10
    do
        echo Running Size:$size Trial:$run
        resultString=`./n-body_simulation $size $steps | grep ms | egrep -o '[0-9]+.[0-9]+'`
        #echo $resultString
        readarray -t results <<<$resultString
        echo ${results[0]} ${results[1]} ${results[2]}
        if [[ $run -ne 1 ]]; then
            echo -n "," >> $cpu_file
            echo -n "," >> $gpu_file
            echo -n "," >> $nt_file
        fi
        echo -n "${results[0]}" >> $cpu_file
        echo -n "${results[1]}" >> $nt_file
        echo -n "${results[2]}" >> $gpu_file
    done
    cat $cpu_file >> $end_file
    echo -ne "\n" >> $end_file
    cat $nt_file >> $end_file
    echo -ne "\n" >> $end_file
    cat $gpu_file >> $end_file
done
