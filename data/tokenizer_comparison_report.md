# Tokenizer Comparison Report
Trajectories compared: 200

Structural (presence/absence) mismatches: 0
PASS: identical presence/absence pattern under both tokenizers, confirming the tokenizer swap affects only magnitude, not extraction logic.

## Magnitude comparison (length predictors)
  mean_memory_length: n=185, Pearson r=0.9972, mean(tiktoken/diagnostic ratio)=1.318
  mean_reflection_length: n=190, Pearson r=0.9641, mean(tiktoken/diagnostic ratio)=1.182
  mean_plan_length: n=192, Pearson r=0.9797, mean(tiktoken/diagnostic ratio)=1.174
