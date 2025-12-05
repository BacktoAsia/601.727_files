# Agentic Reasoning: Reasoning LLM with Agentic Tools
An open-source framework for deep research and beyond. The core idea is to integrate agentic tools into LLM reasoning.

Warning: Still in development. Theoretically runnable, but undergoing rapid updates.

## Install
install from environment.yml, e.g.
```
conda env create -f environment.yml
```

## Run
export your remote LLM API key in environment if you are using remote LLM, e.g.
```
export OPENAI_API_KEY="your openai api key"
```

export your you.com api key in environment if you are using deep research
```
export YDC_API_KEY="your you.com api key"
```

prepare JINA and BING API key if needed

run command:
```
python scripts/run_agentic_reason.py \
--use_jina True \
--jina_api_key "your jina api key" \
--bing_subscription_key "your bing api key"\ 
--remote_model "your remote model name, e.g. gpt-4o" \
--mind_map True \ (optional)
--deep_research True \ (optional, if you want to use deep research)
```

## TODO LIST
- [ ] auto research
- [ ] clean agentic reasoning


## Thanks
Code copied a lot from ...

## Ref
~~~
@misc{wu2025agenticreasoningreasoningllms,
      title={Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research}, 
      author={Junde Wu and Jiayuan Zhu and Yuyuan Liu},
      year={2025},
      eprint={2502.04644},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.04644}, 
}
~~~

