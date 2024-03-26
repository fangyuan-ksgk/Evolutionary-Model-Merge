# Evolutionary-Model-Merge
Unofficial Implementation of Evolutionary Model Merging
* Inspired by work from Sakana.AI [Evolutionary Optimization of Model Merging Recipes] (https://arxiv.org/abs/2403.13187)
* Built with reference on code from Maxime Labonne's code in AutoMerger, MergeKit, and of course, CLAUDE-3
* Computation done on @Modal

https://github.com/fangyuan-ksgk/Evolutionary-Model-Merge/assets/66006349/9750ba46-61ac-4605-9c67-ef1cfe8fc551

![image](https://github.com/fangyuan-ksgk/Evolutionary-Model-Merge/assets/66006349/2b758f02-b5d1-4a41-8897-217021b8fa50)

To run your own evolutionary model merge optimizer, simply use
```
python evolve.py
```
Evaluating fitness score of a LLM is done by computing the average perplexity score on a instruction-following dataset. I use a experimental one [Ksgk-fy/alignment-sft-test01]. Feel free to replace that with yours ;> Following code allows one to evaluate the model's performance. 
```
modal run eval.py --model-id Mistral-7B-Instruct-v0.2
```
Model Merging with config is done through
```
modal run merge.py --unique_id
```
As a experimental run, I've only scraped the top 2 performing 7B LLM from the open llm leaderboard, and SLERP merging is carried out only.




