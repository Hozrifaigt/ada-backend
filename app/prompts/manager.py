import yaml
import logging
from pathlib import Path
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Loads and serves versioned, templatized prompts from a YAML file.
    """
    def __init__(self, prompt_filepath: str):
        self.filepath = Path(prompt_filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Prompt file not found at: {self.filepath}")
        
        self._prompts = self._load_prompts()
        logger.info(f"PromptManager loaded {len(self._prompts)} prompts from {self.filepath.name}")

    def _load_prompts(self):
        with open(self.filepath, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, name: str) -> ChatPromptTemplate:
        prompt_config = self._prompts.get(name)
        if not prompt_config:
            raise ValueError(f"Prompt '{name}' not found in {self.filepath.name}")
        
        template = prompt_config.get("template")
        if not template:
            raise ValueError(f"Prompt '{name}' in {self.filepath.name} is missing a 'template' key.")
        
        return ChatPromptTemplate.from_template(template)