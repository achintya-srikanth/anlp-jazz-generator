# TO TRAIN
python3 train.py ckpt_dir log_file

# TO TRAIN WITH SLURM
sbatch submit_train.job
squeue -u gzhang7

# TO INFERENCE
python3 inference.py --model ckpt_dir/model-055-0.798 metrics/MusDr/music/music_midi/output055.midi
python3 inference.py --model ckpt_dir/model-271-0.237 metrics/MusDr/music/music_midi/output271.midi

# TO INFERENCE 10 TIMES
for i in {1..10}; do python3 inference.py --model ckpt_dir/model-055-0.798 metrics/MusDr/music/music_midi/output055_$i.midi; done
for i in {1..10}; do python3 inference.py --model ckpt_dir/model-271-0.237 metrics/MusDr/music/music_midi/output271_$i.midi; done

# MOVE GENERATED CSV TO music/music_symbolic
mv /ocean/projects/cis250010p/gzhang7/jazz_transformer/metrics/MusDr/music/music_midi/*.csv /ocean/projects/cis250010p/gzhang7/jazz_transformer/metrics/MusDr/music/music_symbolic/


# DOWNLOAD
fluidsynth -ni "FluidR3_GM.sf2" NAME.midi -F NAME.wav  # LOCAL

for f in in *.midi; do fluidsynth -ni "~/Desktop/FluidR3_GM.sf2" "$f" -F "../wav_files/${f%.*}.wav"; done
# UPLOAD


# METRICS ON GENERATED
conda activate metric_env
cd metrics/MusDr/

python run_python_scapeplot.py \
    -a music/music_a  \
    -s music/music_ssm   \
    -p music/music_p

python run_all_metrics.py \
    -s music/music_symbolic  \
    -p music/music_p  \
    -o music/all_metrics.csv

# VISUALIZATION
python3 vis_scapeplot.py \
    -p music/music_p  \
    -f music/music_fig







# METRICS ON TEST DATA
python run_python_scapeplot.py \
    -a musdr/testdata/audio  \
    -s musdr/testdata/ssm   \
    -p musdr/testdata/scplot
python run_all_metrics.py \
    -s musdr/testdata/symbolic  \
    -p musdr/testdata/scplot
#-o testout.csv




