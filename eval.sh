
cmd='
cd /lustre/fsw/portfolios/llmservice/users/sdiao/hymba/lm-evaluation-harness;
pip install -e .[math,ifeval,sentencepiece]
which python;
which pip;
pip list;

lm_eval --model hf --model_args pretrained=nvidia/Hymba-1.5B-Instruct,dtype=bfloat16,trust_remote_code=True \
     --tasks mmlu,gsm8k \
     --num_fewshot 5 \
     --batch_size 1 \
     --output_path /lustre/fsw/portfolios/llmservice/users/sdiao/hymba/lm-evaluation-harness/lm-results \
     --log_samples \
     --use_cache /lustre/fsw/portfolios/llmservice/users/sdiao/hymba/lm-evaluation-harness/hymba_cache \
     --apply_chat_template --fewshot_as_multiturn --device cuda:0
      '

cmd2='
cd /lustre/fsw/portfolios/llmservice/users/sdiao/hymba/lm-evaluation-harness;
pip install -e .[math,ifeval,sentencepiece]
which python;
which pip;
pip list;

lm_eval --model hf --model_args pretrained=nvidia/Hymba-1.5B-Instruct,dtype=bfloat16,trust_remote_code=True \
     --tasks ifeval,gpqa \
     --num_fewshot 0 \
     --batch_size 1 \
     --output_path /lustre/fsw/portfolios/llmservice/users/sdiao/hymba/lm-evaluation-harness/lm-results \
     --log_samples \
     --use_cache /lustre/fsw/portfolios/llmservice/users/sdiao/hymba/lm-evaluation-harness/hymba_cache \
     --apply_chat_template --fewshot_as_multiturn --device cuda:0
      '

submit_job --gpu 1 --nodes 1 -n lm_eval --notify_on_start --noroot --partition batch_short,backfill,batch --duration 2 --mounts $HOME:/home/sdiao,/lustre:/lustre,/usr/bin/pdsh:/usr/bin/pdsh,/usr/lib/pdsh:/usr/lib/pdsh --image /lustre/fsw/portfolios/llmservice/users/sdiao/docker/megatron_py25.sqsh --command ''"${cmd}"''
submit_job --gpu 1 --nodes 1 -n lm_eval --notify_on_start --noroot --partition batch_short,backfill,batch --duration 2 --mounts $HOME:/home/sdiao,/lustre:/lustre,/usr/bin/pdsh:/usr/bin/pdsh,/usr/lib/pdsh:/usr/lib/pdsh --image /lustre/fsw/portfolios/llmservice/users/sdiao/docker/megatron_py25.sqsh --command ''"${cmd2}"''


