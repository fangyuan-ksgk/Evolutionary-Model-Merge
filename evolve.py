import random
import subprocess
import numpy as np
import yaml

# The idea is that in merge.py we define a fixed 2-model merging process
# And in evolve.py we define a function that evolves the merging configuration (between the 2 best-performing models)

get_slerp_config = lambda w: f"""
slices:
  - sources:
      - model: liminerity/M7-7b
        layer_range: [0, 32]
      - model: AurelPx/Percival_01-7b-slerp
        layer_range: [0, 32]
merge_method: slerp
base_model: liminerity/M7-7b
parameters:
  t:
    - filter: self_attn
      value: [{w[0]}, {w[1]}, {w[2]}, {w[3]}, {w[4]}]
    - filter: mlp
      value: [{1-w[0]}, {1-w[1]}, {1-w[3]}, {1-w[3]}, {1-w[4]}]
    - value: {w[5]}
dtype: bfloat16
random_seed: 0
    """

def save_config(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def generate_random_config():
    """
    Slerp configuration generator
    """
    w = [random.uniform(0, 1) for _ in range(6)]
    config = get_slerp_config(w)
    unique_id = '-'.join([str(np.round(x,2)) for x in w])
    print('unique_id:', unique_id)

    # Save the configuration to a YAML file
    name = f"{unique_id}"
    local_file_path = f"./config/{name}.yaml"
    save_config(config, local_file_path)

    return name

def read_model_name():
    try:
        with open("./merge_info/model_name.txt", "r") as f:
            model_name = f.read().strip()
        return model_name
    except FileNotFoundError:
        print("Model name file not found.")
        return None



def evaluate_config(unique_id):
    # Merge according to the configuration, Evaluate the merged model with instruction dataset
    command = "ls"
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    print("List current directory: ", result.stdout)
    # Evaluation requires running the merging process & Evaluation script
    # merge: modal run merge.py --config yaml_config 
    # note that the yaml_config is the string config that we have above
    # TBD: addition of unique_id to create new model name
    command = ["modal", "run", "merge.py", "--unique-id", unique_id]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    print("=================================")
    print("         Merging Complete")
    print("=================================")

    model_name = read_model_name()
    print(f"Model name: {model_name}")

    # Evaluate the model
    # eval: modal run eval.py --model_name model_name
    # note that model_name is user_name/model_name in fact
    hf_user_name = "Ksgk-fy"
    command = ["modal", "run", "eval.py", "--model-name", f"{hf_user_name}/{model_name}"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    score = random.random()  # Placeholder for demonstration
    return score

# Honestly I feel like a LLM might have a better intuition on where things should be going, but this is also fine (??)
# the random spawning process looks a bit dumb, plus, how come there are only two models getting merged here???
# fix it, think about it!

def evolve_configs(population_size, generations):
    population = [generate_random_config() for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [evaluate_config(unique_id) for unique_id in population]

        # Select the best configurations based on fitness scores
        best_configs = [config for _, config in sorted(zip(fitness_scores, population), reverse=True)][:population_size//2]

        # Generate new configurations by mutating the best ones
        new_configs = []
        for config in best_configs:
            new_config = yaml.safe_load(yaml.dump(config))  # Deep copy of config
            # Mutate the new configuration
            if random.random() < 0.5:
                model_index = random.randint(1, len(new_config['models'])-1)
                new_config['models'][model_index]['parameters']['density'] = random.uniform(0.1, 0.9)
                new_config['models'][model_index]['parameters']['weight'] = random.uniform(0.1, 0.9)
            new_configs.append(new_config)

        population = best_configs + new_configs

    best_config = max(population, key=evaluate_config)
    return best_config

# Make sure direcroty exists | Copy to cloud storage
import os
os.makedirs('./config', exist_ok=True)

# Run the evolutionary search
population_size = 10
generations = 50
best_config = evolve_configs(population_size, generations)

# Print the best configuration
print(yaml.dump(best_config))
