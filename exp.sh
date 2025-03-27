rm -rf /dev/shm/out
rm -rf ./out
for model_type in bert whisper
do
  for dataset in Ego4D Youtube
  do
    for strategy in "Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence"
    do
      for seed in 13 42 87
      do
        python3.10 ${model_type}/main.py --dataset ${dataset} --strategy="${strategy}" --seed ${seed}
      done
    done
  done
done
cp -r /dev/shm/out/ ./