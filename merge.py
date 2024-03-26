import modal, yaml
from modal import Image, Stub, gpu
from huggingface_hub import login
import os

HF_TOKEN = "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"
login(HF_TOKEN)

stub = modal.Stub(
    image = Image.debian_slim(python_version="3.10")
    .pip_install(
        ["huggingface_hub", "torch", "tqdm"]
    )
    .apt_install("git")
    .apt_install( "gcc")
    .run_commands(f"export HF_TOKEN={HF_TOKEN}")
    .run_commands("git config --global user.name ksgk-fangyuan",
                  "git config --global user.email fangyuan.yu18@gmail.com",
                  )
    .run_commands("mkdir -p /root")
    .run_commands("ls")
    .run_commands("pwd")
    .run_commands("cd /root && git clone https://github.com/cg123/mergekit.git && cd mergekit && pip install -e .")
    .run_commands("cd /root && git clone https://github.com/Weyaxi/scrape-open-llm-leaderboard && pip install -r ./scrape-open-llm-leaderboard/requirements.txt")
    .run_commands("ls")
    .run_commands("pwd")
    .copy_local_dir(local_path="./config", remote_path="/root/config")
    .run_commands("ls /root/config")
)
# Lesson learned 1: .run_commands is done in the main diretory, which contains /root and /data
# however, the stub.function is executed in /root directory, so git clone should be done in /root
# Lesson learned 2: multiple stub.function do not share the same environment, so files created in one function disappears in another function's environment
import os, re, sys, time, random, yaml, subprocess, shutil, requests
from io import StringIO
import pandas as pd
from jinja2 import Template
from huggingface_hub import ModelCard, ModelCardData, HfApi, repo_info
from huggingface_hub.utils import RepositoryNotFoundError

USERNAME = 'Ksgk-fy'
N_ROWS = 20
WAIT_TIME = 10800
    

def create_dataset() -> bool:
  """
  Use Scrape Open LLM Leaderboard to create a CSV dataset
  """
  command = ["pwd"]
  result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
  print("Current working directory: ", result.stdout)
  command = ["ls"]
  result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
  print("List working directory: ", result.stdout)

  command = ["python3", "scrape-open-llm-leaderboard/main.py", "-csv"]
  
  try:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    print(f"scrape-open-llm-leaderboard: {result.stdout}")
    
  
    command = ["ls"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    print("List working directory: ", result.stdout)
    return True

  except subprocess.CalledProcessError as e:
    print(f"scrape-open-llm-leaderboard: {e.stderr}")
    return False
  
# My feeling is that such 7.24 billion parameter model filter gets us the same model structure
def make_df(file_path: str, n_rows: int) -> pd.DataFrame:
    """
    Create a filtered dataset from the Open LLM Leaderboard.
    """
    import subprocess
    result = subprocess.run(["pwd"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Make Dataframe -- Working directory: ", result.stdout)
    result = subprocess.run(["ls"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("List directory: ", result.stdout)
    result = subprocess.run(["ls", "./config"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("List ./config directory: ", result.stdout)

    columns = ["Available on the hub", "Model sha", "T", "Type", "Precision",
              "Architecture", "Weight type", "Hub ❤️", "Flagged", "MoE"]
    ds = pd.read_csv(file_path)
    df = (
          ds[
            (ds["#Params (B)"] == 7.24) &
            (ds["Available on the hub"] == True) &
            (ds["Flagged"] == False) &
            (ds["MoE"] == False) &
            (ds["Weight type"] == "Original")
          ]
          .drop(columns=columns)
          .drop_duplicates(subset=["Model"])
          .iloc[:n_rows]
      )
    return df

def repo_exists(repo_id: str) -> bool:
    try:
        repo_info(repo_id)
        return True
    except RepositoryNotFoundError:
        return False
    
def get_name(models: list[pd.Series], username: str, version=0, unique_id: str="") -> str:
    model_name = models[0]["Model"].split("/")[-1].split("-")[0].capitalize() \
                 + models[1]["Model"].split("/")[-1].split("-")[0].capitalize() \
                 + unique_id + "-7B" 
    if version > 0:
        model_name = model_name.split("-")[0] + unique_id + f"-v{version}-7B"

    if repo_exists(f"{username}/{model_name}"):
        get_name(models, username, version+1)

    return model_name

def get_license(models: list[pd.Series]) -> str:
    license1 = models[0]["Hub License"]
    license2 = models[1]["Hub License"]
    license = "cc-by-nc-4.0"

    if license1 == "cc-by-nc-4.0" or license2 == "cc-by-nc-4.0":
        license = "cc-by-nc-4.0"
    elif license1 == "apache-2.0" or license2 == "apache-2.0":
        license = "apache-2.0"
    elif license1 == "MIT" and license2 == "MIT":
        license = "MIT"
    return license

def create_config(models: list[pd.Series]) -> str:
    slerp_config = f"""
slices:
  - sources:
      - model: {models[0]["Model"]}
        layer_range: [0, 32]
      - model: {models[1]["Model"]}
        layer_range: [0, 32]
merge_method: slerp
base_model: {models[0]["Model"]}
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
random_seed: 0
    """

    dare_config = f"""
models:
- model: {models[0]["Model"]}
  # No parameters necessary for base model
- model: {models[1]["Model"]}
  parameters:
    density: 0.53
    weight: 0.6
merge_method: dare_ties
base_model: {models[0]["Model"]}
parameters:
int8_mask: true
dtype: bfloat16
random_seed: 0
"""
    yaml_config = random.choices([slerp_config, dare_config], weights=[0.4, 0.6], k=1)[0]

    with open('config.yaml', 'w', encoding="utf-8") as f:
        f.write(yaml_config)

    return yaml_config

def download_leaderboard():
    """
    Download the gist that contains the leaderboard.
    """
    url = "https://gist.githubusercontent.com/automerger/84af749b1c0ef7336858df408f46f388/raw"
    file_path = "leaderboard.txt"
    response = requests.get(url)
    return response.content.decode('utf-8')

def convert_markdown_table_to_dataframe(md_content):
    """
    Converts markdown table to Pandas DataFrame.
    """
    # Remove leading and trailing | characters
    cleaned_content = re.sub(r'\|\s*$', '', re.sub(r'^\|\s*', '', md_content, flags=re.MULTILINE), flags=re.MULTILINE)

    # Create DataFrame from cleaned content
    df = pd.read_csv(StringIO(cleaned_content), sep="\|", engine='python')

    # Remove the first row after the header
    df = df.drop(0, axis=0)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    return df

def get_dataframe():
    """
    Wrapper to update the Gradio dataframe.
    """
    content = download_leaderboard()
    df = convert_markdown_table_to_dataframe(content)
    return df

def clear_data():
    """
    Clear data so the Space doesn't crash...
    """
    dir_path = "/data/merge"
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"The directory '{dir_path}' has been removed successfully.")
    else:
        print(f"The directory '{dir_path}' does not exist.")

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


  
# This function requires local access to the config.yaml, wonder if we can just modal that
def merge_models() -> None:
  """
  Use mergekit to create a merge
  """
  command = ["mergekit-yaml", "config.yaml", "/data/merge", "--copy-tokenizer"]

  with open("output.log", "a") as log_file:
    try:
      result = subprocess.run(command, check=True, stdout=log_file,
                              stderr=log_file, text=True)
      print(f"mergekit: {result.stdout}")
    except subprocess.CalledProcessError as e:
      print(f"mergekit: {e.stderr}")

def create_model_card(yaml_config: str, model_name: str, username: str, license: str) -> None:
    template_text = """
---
license: {{ license }}
base_model:
{%- for model in models %}
  - {{ model }}
{%- endfor %}
tags:
- merge
- mergekit
- lazymergekit
- automerger
---

## 🧩 Configuration
```yaml
{{- yaml_config -}}
```
## 💻 Usage
```python
!pip install -qU transformers accelerate
from transformers import AutoTokenizer
import transformers
import torch
model = "{{ username }}/{{ model_name }}"
messages = [{"role": "user", "content": "What is a large language model?"}]
tokenizer = AutoTokenizer.from_pretrained(model)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```
"""

    # Create a Jinja template object
    jinja_template = Template(template_text.strip())

    # Get list of models from config
    data = yaml.safe_load(yaml_config)
    if "models" in data:
        models = [data["models"][i]["model"] for i in range(len(data["models"])) if "parameters" in data["models"][i]]
    elif "parameters" in data:
        models = [data["slices"][0]["sources"][i]["model"] for i in range(len(data["slices"][0]["sources"]))]
    elif "slices" in data:
        models = [data["slices"][i]["sources"][0]["model"] for i in range(len(data["slices"]))]
    else:
        raise Exception("No models or slices found in yaml config")

    # Fill the template
    content = jinja_template.render(
        model_name=model_name,
        models=models,
        yaml_config=yaml_config,
        username=username,
        license=license
    )

    # Save the model card
    card = ModelCard(content)
    card.save('/data/merge/README.md')

def upload_model(api: HfApi, username: str, model_name: str) -> None:
    """
    Upload merged model to the Hugging Face Hub.
    """
    api.create_repo(
        repo_id=f"{username}/{model_name}",
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=f"{username}/{model_name}",
        folder_path="/data/merge",
    )

def load_config(file_path):
    # Read the YAML file
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

@stub.function(cpu=8.0, memory=262144, timeout=1200)
def merge_models_with_config(unique_id: str = "") -> str:
    """
    Merge models based on the given configuration.
    """
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/merge"):
        os.makedirs("data/merge")
    
    api = HfApi(token=HF_TOKEN)

    create_dataset()
    df = make_df("./open-llm-leaderboard.csv", N_ROWS)
    

    # Sample two models
    dir_path = "/data"
    sample = df[:2]
    models = [sample.iloc[i] for i in range(2)]

    # Get model name
    model_name = get_name(models, USERNAME, version=0, unique_id=unique_id)
    print("="*60)
    print(f"Model name: {model_name}")

    # Get model license
    license = get_license(models)
    print(f"License: {license}")

    # Merge configs
    if unique_id == "":
        print("Create new config")
        yaml_config = create_config(models)
    else:
        print("Load existing config")
        yaml_config = load_config(f"./config/{unique_id}.yaml")

    # save the config in ./config.yaml | rewrite if already exists
    with open('config.yaml', 'w', encoding="utf-8") as f:
        f.write(yaml_config)

    print(f"YAML config:{yaml_config}")
    print(f"Data size: {human_readable_size(get_size(dir_path))}")

    # Merge models
    merge_models()
    print("Model merged!")

    # Create model card
    print("Create model card")
    create_model_card(yaml_config, model_name, USERNAME, license)

    # Upload model
    print("Upload model")
    upload_model(api, USERNAME, model_name)


    return model_name


@stub.local_entrypoint()
def main(unique_id: str = ""):
    model_name = merge_models_with_config.remote(unique_id)

    # Save the model name locally
    with open("./merge_info/model_name.txt", "w") as f:
        f.write(model_name)

    return model_name
        
    


