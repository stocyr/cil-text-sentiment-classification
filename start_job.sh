bsub -n 8 -W 3:00 -N -B -R "rusage[mem=2048]" < main.sh
