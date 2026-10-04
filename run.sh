#!/usr/bin/env bash
# AdvSim2Real launcher:  bash run.sh stage1 | stage2 | eval | all
#   stage1  Algorithm 1: curriculum <-> executor, T1 rounds        -> runs_stage1/{curr_v*, exec_v*}
#   stage2  Algorithm 2: adversary <-> executor, curriculum frozen -> runs_stage2/{adv_v*, exec_v*}
#   eval    clean + reactive-attack completion on data/tasks.json  -> eval/*.json
# WebWorld-14B must already be served at $WORLD_API. Every setting below is an environment variable.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}; VLLM=${VLLM:-vllm}; BASE=${BASE:-Qwen/Qwen3.5-4B}
WORLD_NAME=${WORLD_NAME:-WebWorld-14B}; WORLD_API=${WORLD_API:-http://127.0.0.1:8004/v1}
EXEC_PORT=${EXEC_PORT:-8005}; SECOND_PORT=${SECOND_PORT:-8006}
EXEC_API="http://127.0.0.1:$EXEC_PORT/v1"; SECOND_API="http://127.0.0.1:$SECOND_PORT/v1"
TGPU=${TGPU:-3}; EXEC_SERVE_GPU=${EXEC_SERVE_GPU:-0}; SECOND_SERVE_GPU=${SECOND_SERVE_GPU:-2}; GMU=${GMU:-0.4}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.8-27B}; JUDGE_WORKERS=${JUDGE_WORKERS:-16}
LR=${LR:-1e-5}; KL_BETA=${KL_BETA:-1e-2}; REP_LAMBDA=${REP_LAMBDA:-0.3}; MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}; PPS=${PPS:-2}
T1=${T1:-3}; T2=${T2:-3}; S1=${S1:-runs_stage1}; S2=${S2:-runs_stage2}
TRAIN=(--attn_impl sdpa --grad_checkpoint 1 --lr "$LR" --kl_beta "$KL_BETA")
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} VLLM_NO_USAGE_STATS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SP=""; EP=""
serve() {  # port gpu log alias [lora_dir] -> prints pid
  local port=$1 gpu=$2 log=$3 alias=$4 adapter=${5:-} pid
  local extra=(--served-model-name "$alias")
  [ -z "$adapter" ] || extra=(--enable-lora --lora-modules "$alias=$adapter" --max-lora-rank 32)
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$BASE" --host 127.0.0.1 --port "$port" "${extra[@]}" \
    --max-model-len 16384 --gpu-memory-utilization "$GMU" --enforce-eager >"$log" 2>&1 &
  pid=$!
  for _ in $(seq 360); do
    curl -sf "http://127.0.0.1:$port/v1/models" | grep -q "\"$alias\"" && { echo "$pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || { tail -20 "$log" >&2; return 1; }
    sleep 5
  done
  return 1
}
stop() {  # pid port
  [ -z "${1:-}" ] || kill -9 -- "-$1" 2>/dev/null || kill -9 "$1" 2>/dev/null || true
  for _ in $(seq 60); do curl -s "http://127.0.0.1:$2/v1/models" >/dev/null || return 0; sleep 1; done
}
trap 'stop "$SP" $SECOND_PORT; stop "$EP" $EXEC_PORT' EXIT
judge() {  # tasks out K max_steps [extra...]
  "$PY" -m advsim2real.judge.judge_rollouts --tasks "$1" --out "$2" --K "$3" --max_steps "$4" --executor_api "$EXEC_API" \
    --world_api "$WORLD_API" --world_model "$WORLD_NAME" --judge_model "$JUDGE_MODEL" --workers "$JUDGE_WORKERS" "${@:5}"
}
propose() {  # out n
  "$PY" -m advsim2real.curriculum.propose --out "$1" --n_proposals "$2" --curriculum_api "$SECOND_API"
}

stage1() {   # defaults: recorded Stage-1 configuration (paper App. C.6)
  local NPROP=${NPROP:-150} K=${K:-6} MAX_STEPS=${MAX_STEPS:-8} init_e=() init_c=() e="" c=""
  mkdir -p "$S1"
  for t in $(seq 1 "$T1"); do
    if [ "$t" -gt 1 ]; then e="$S1/exec_v$((t-1))"; c="$S1/curr_v$((t-1))"; init_e=(--init_adapter "$e"); init_c=(--init_adapter "$c"); fi
    # A) curriculum proposes, executor assessed, curriculum update
    SP=$(serve $SECOND_PORT $SECOND_SERVE_GPU "$S1/curr_serve_v$t.log" curr "$c")
    EP=$(serve $EXEC_PORT $EXEC_SERVE_GPU "$S1/exec_serve_v$t.log" exec "$e")
    propose "$S1/raw_tasks_v$t.json" "$NPROP"
    judge "$S1/raw_tasks_v$t.json" "$S1/judged_tasks_v$t.json" "$K" "$MAX_STEPS"
    stop "$SP" $SECOND_PORT; stop "$EP" $EXEC_PORT; SP=""; EP=""
    CUDA_VISIBLE_DEVICES=$TGPU "$PY" -m advsim2real.curriculum.train --model "$BASE" ${init_c[@]+"${init_c[@]}"} \
      --tasks_file "$S1/judged_tasks_v$t.json" --out "$S1/curr_v$t" --epochs "${CURR_EPOCHS:-1}" \
      --group_size "${CURR_GROUP:-6}" --prompts_per_step "$PPS" --rep_lambda "$REP_LAMBDA" --micro_bs 1 "${TRAIN[@]}"
    # B) updated curriculum proposes, executor assessed, executor update
    SP=$(serve $SECOND_PORT $SECOND_SERVE_GPU "$S1/curr_serveB_v$t.log" curr "$S1/curr_v$t")
    EP=$(serve $EXEC_PORT $EXEC_SERVE_GPU "$S1/exec_serveB_v$t.log" exec "$e")
    propose "$S1/raw_exec_tasks_v$t.json" "$NPROP"
    judge "$S1/raw_exec_tasks_v$t.json" "$S1/judged_exec_tasks_v$t.json" "$K" "$MAX_STEPS"
    stop "$SP" $SECOND_PORT; stop "$EP" $EXEC_PORT; SP=""; EP=""
    CUDA_VISIBLE_DEVICES=$TGPU "$PY" -m advsim2real.executor.train --model "$BASE" ${init_e[@]+"${init_e[@]}"} \
      --tasks_file "$S1/judged_exec_tasks_v$t.json" --out "$S1/exec_v$t" --epochs "${EXEC_EPOCHS:-2}" \
      --group_size "${EXEC_GROUP:-6}" --prompts_per_step "$PPS" --max_steps "$MAX_STEPS" \
      --world_api "$WORLD_API" --world_model "$WORLD_NAME" --judge_model "$JUDGE_MODEL" \
      --max_seq_len "$MAX_SEQ_LEN" "${TRAIN[@]}"
  done
}

stage2() {
  local NPROP=${NPROP:-40} K=${K:-4} MAX_STEPS=${MAX_STEPS:-8} MIN=${MIN_CLEAN_SUCCESS:-0.5}
  local curr="$S1/curr_v$T1" exe adv_init history
  mkdir -p "$S2"
  for t in $(seq 1 "$T2"); do
    exe="$S1/exec_v$T1"; adv_init=()
    if [ "$t" -gt 1 ]; then exe="$S2/exec_v$((t-1))"; adv_init=(--init_adapter "$S2/adv_v$((t-1))"); fi
    # 1) frozen curriculum proposes; current executor assessed clean -> attackable tasks
    SP=$(serve $SECOND_PORT $SECOND_SERVE_GPU "$S2/curr_serve_v$t.log" curr "$curr")
    propose "$S2/base_tasks_v$t.json" "$NPROP"
    stop "$SP" $SECOND_PORT; SP=""
    EP=$(serve $EXEC_PORT $EXEC_SERVE_GPU "$S2/exec_serve_v$t.log" exec "$exe")
    judge "$S2/base_tasks_v$t.json" "$S2/judged_clean_v$t.json" "$K" "$MAX_STEPS" --max_repeat 0
    "$PY" -m advsim2real.adversary.stage2_data prepare --tasks "$S2/judged_clean_v$t.json" --out "$S2/eligible_base_v$t.json" \
      --clean_out "$S2/clean_tasks_v$t.json" --min_clean_success "$MIN"
    # 2) adversary update against the fixed executor (success-flip reward)
    CUDA_VISIBLE_DEVICES=$TGPU "$PY" -m advsim2real.adversary.train --model "$BASE" ${adv_init[@]+"${adv_init[@]}"} \
      --base_tasks "$S2/eligible_base_v$t.json" --min_clean_success "$MIN" --K "$K" --max_steps "$MAX_STEPS" \
      --exec_api "$EXEC_API" --world_api "$WORLD_API" --world_model "$WORLD_NAME" --judge_model "$JUDGE_MODEL" \
      --judge_workers "$JUDGE_WORKERS" --out "$S2/adv_v$t" --steps "${ADV_STEPS:-6}" --epochs "${ADV_EPOCHS:-0}" \
      --group_size "${ADV_GROUP:-6}" --prompts_per_step "$PPS" --rep_lambda "$REP_LAMBDA" --micro_bs 1 "${TRAIN[@]}"
    # 3) attacks from adv_v$t + earlier attacks + clean tasks, assessed with the current executor
    SP=$(serve $SECOND_PORT $SECOND_SERVE_GPU "$S2/adv_serve_v$t.log" adv "$S2/adv_v$t")
    "$PY" -m advsim2real.adversary.build_tasks --pool "$S2/eligible_base_v$t.json" --out "$S2/raw_adv_v$t.json" \
      --adversary_api "$SECOND_API" --seed "$t"
    stop "$SP" $SECOND_PORT; SP=""
    history=(); for p in $(seq 1 $((t-1))); do history+=("$S2/raw_adv_v$p.json"); done
    "$PY" -m advsim2real.adversary.stage2_data mix --fresh "$S2/raw_adv_v$t.json" --clean "$S2/clean_tasks_v$t.json" \
      --history ${history[@]+"${history[@]}"} --out "$S2/raw_exec_mix_v$t.json" --seed "$t" \
      --replay_fraction "${REPLAY_FRACTION:-0.25}" --clean_fraction "${CLEAN_FRACTION:-0.25}"
    judge "$S2/raw_exec_mix_v$t.json" "$S2/judged_exec_mix_v$t.json" "$K" "$MAX_STEPS" --max_repeat 0
    stop "$EP" $EXEC_PORT; EP=""
    # 4) executor update on the mixture
    CUDA_VISIBLE_DEVICES=$TGPU "$PY" -m advsim2real.executor.train --model "$BASE" --init_adapter "$exe" \
      --tasks_file "$S2/judged_exec_mix_v$t.json" --out "$S2/exec_v$t" --steps "${EXEC_STEPS:-6}" \
      --epochs "${EXEC_EPOCHS:-1}" --group_size "${EXEC_GROUP:-4}" --prompts_per_step "$PPS" --max_steps "$MAX_STEPS" \
      --world_api "$WORLD_API" --world_model "$WORLD_NAME" --judge_model "$JUDGE_MODEL" \
      --max_seq_len "$MAX_SEQ_LEN" "${TRAIN[@]}"
  done
}

evaluate() {  # EXECS="tag:adapter_dir ..." (empty dir = base model); EVAL_ADV = attacker adapter
  local out=${OUT:-eval} tag adapter
  mkdir -p "$out"
  SP=$(serve $SECOND_PORT $SECOND_SERVE_GPU "$out/adv_serve.log" adv "${EVAL_ADV:-$S2/adv_v$T2}")
  for entry in ${EXECS:-"base: s1:$S1/exec_v$T1 s2:$S2/exec_v$T2"}; do
    tag=${entry%%:*}; adapter=${entry#*:}
    EP=$(serve $EXEC_PORT $EXEC_SERVE_GPU "$out/${tag}_serve.log" exec "$adapter")
    for mode in clean attacked; do
      "$PY" -m advsim2real.evaluation.robustness --bench data/tasks.json --mode "$mode" --adversary_api "$SECOND_API" \
        --executor_api "$EXEC_API" --world_api "$WORLD_API" --world_model "$WORLD_NAME" --seeds "${SEEDS:-0 1 2}" \
        --max_steps "${EVAL_MAX_STEPS:-12}" --temperature "${TEMP:-0.7}" --judge_model "$JUDGE_MODEL" \
        --workers "$JUDGE_WORKERS" --out "$out/${tag}_${mode}.json"
    done
    stop "$EP" $EXEC_PORT; EP=""
  done
}

curl -sf "$WORLD_API/models" >/dev/null || { echo "ERROR: WebWorld is not answering at $WORLD_API"; exit 1; }
case "${1:-}" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  eval)   evaluate ;;
  all)    stage1; stage2; evaluate ;;
  *) echo "usage: bash run.sh stage1|stage2|eval|all"; exit 1 ;;
esac
