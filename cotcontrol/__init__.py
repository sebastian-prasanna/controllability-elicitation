"""CoT-Control-QA eval package: OpenRouter inference, prompts, grading, eval loop."""

from cotcontrol.cotcontrol_eval import CONSTRAINT_MODES, eval_cotcontrolqa  # noqa: F401
from cotcontrol.or_inference import GenerateConfig, generate_async, get_client  # noqa: F401
