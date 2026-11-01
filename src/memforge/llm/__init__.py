"""Structured LLM integration helpers.

LiteLLM reads its model table, the limits and capabilities of every model it
knows, when it is first imported. MemForge reads the copy packaged with the
LiteLLM version it pins, so a model's limits change only with a LiteLLM
upgrade and no process fetches the table from the network. A model newer than
that copy runs with the token limits the operator configures.
"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
