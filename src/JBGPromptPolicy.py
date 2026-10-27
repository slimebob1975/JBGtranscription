"""
The prompt policy: the policy file, the summary options and the standard
texts of the editable instructions.

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import json
from pathlib import Path
from src.JBGLogger import JBGLogger

logger = JBGLogger(level="INFO").logger


# The fixed opening of the custom analysis, used if the policy file lacks it.
DEFAULT_CUSTOM_ANALYSIS_PREAMBLE = (
    "Du analyserar och sammanställer transkriberad text från en ljudinspelning. "
    "Gör följande analys:"
)
# Upper bound on a user-supplied summary instruction.
MAX_SUMMARY_PROMPT_CHARS = 20000
# Option ids used before the summary options were made configurable. Browsers
# still hold these in localStorage, and older clients still post them, so they
# are mapped onto their current equivalents rather than rejected.
LEGACY_SUMMARY_OPTION_IDS = {
    "short": "enkel",
    "extensive": "utforlig",
}
# Used only if the prompt policy cannot be read or contains no usable options.
FALLBACK_SUMMARY_OPTION_ID = "standard"
FALLBACK_SUMMARY_PROMPT = (
    "Sammanfatta följande transkribering på ett tydligt och sakligt sätt. "
    "Hitta inte på information. Om något är oklart, skriv att det är oklart."
)


class PromptPolicyMixin:
    # Resolved from the package location rather than the current working
    # directory, so the policy is found regardless of how the app is started.
    POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "prompt_policy.json"

    @staticmethod
    def load_policy_file():
        """Load the prompt policy from disk. Returns an empty dict on failure."""
        try:
            with open(PromptPolicyMixin.POLICY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f" Could not load prompt policy: {e}")
            return {}

    @staticmethod
    def analysis_prompt_defaults(policy=None):
        """Standard texts for the editable analyses, for the GUI."""
        policy = policy if policy is not None else PromptPolicyMixin.load_policy_file()
        return {
            "follow_up_questions": PromptPolicyMixin.policy_text(
                policy, "follow_up_questions", "Generera uppföljningsfrågor."
            ),
            "speaker_diarization": PromptPolicyMixin.policy_text(
                policy, "speaker_diarization",
                "Försök att identifiera olika röster i följande transkribering:",
            ),
            "custom_analysis_preamble": PromptPolicyMixin.policy_text(
                policy, "custom_analysis_preamble", DEFAULT_CUSTOM_ANALYSIS_PREAMBLE
            ),
        }

    @staticmethod
    def policy_text(policy, key, default=""):
        """Return a policy entry as a single string.

        Entries are written either as a plain string or as a list of lines.
        Joining a string with "\\n".join() would insert a newline between every
        character, so the type has to be checked before joining.
        """
        value = policy.get(key, default)
        if isinstance(value, (list, tuple)):
            return "\n".join(str(line) for line in value)
        return str(value)

    @classmethod
    def summary_options(cls, policy=None):
        """Return the selectable summary options, in display order.

        Each option is a dict with id, label, description, default and prompt.
        The list is the single source of truth for both the GUI dropdown and
        the server-side validation of the chosen option.
        """
        policy = policy if policy is not None else cls.load_policy_file()
        raw = policy.get("summary_options") or []

        options = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            option_id = str(entry.get("id", "")).strip()
            prompt = cls.policy_text(entry, "prompt")
            if not option_id or not prompt.strip():
                continue
            options.append({
                "id": option_id,
                "label": str(entry.get("label") or option_id),
                "description": str(entry.get("description") or ""),
                "default": bool(entry.get("default")),
                "prompt": prompt,
            })

        if not options:
            logger.error(
                "No usable summary options in the prompt policy. "
                "Falling back to a single built-in instruction."
            )
            options = [{
                "id": FALLBACK_SUMMARY_OPTION_ID,
                "label": "Standard",
                "description": "",
                "default": True,
                "prompt": FALLBACK_SUMMARY_PROMPT,
            }]

        if not any(o["default"] for o in options):
            logger.warning(
                "No summary option is marked as default; using the first one."
            )
            options[0]["default"] = True

        return options

    @classmethod
    def resolve_summary_option_id(cls, option_id, options=None):
        """Map a requested option id onto an existing one.

        Ids that no longer exist fall back to the default option rather than
        failing, so a stale value stored in a browser does not break the run.
        """
        options = options if options is not None else cls.summary_options()
        known = {o["id"] for o in options}

        candidate = (option_id or "").strip()
        candidate = LEGACY_SUMMARY_OPTION_IDS.get(candidate, candidate)

        if candidate in known:
            return candidate

        default_id = next(o["id"] for o in options if o["default"])
        if candidate:
            logger.warning(
                f"Unknown summary option '{option_id}'; using default '{default_id}'."
            )
        return default_id

    def load_prompt_policy(self):
        return self.load_policy_file()

    def resolve_summary_instructions(self, style=None, custom_prompt=None):
        """Decide which instruction text to use for the summary.

        A non-empty custom prompt from the user always wins. Otherwise the
        instruction belonging to the chosen option is taken from the prompt
        policy, falling back to the default option.
        """
        candidate = (custom_prompt or "").strip()
        if candidate:
            if len(candidate) > MAX_SUMMARY_PROMPT_CHARS:
                logger.warning(
                    f"Summary instruction truncated from {len(candidate)} to "
                    f"{MAX_SUMMARY_PROMPT_CHARS} characters."
                )
                candidate = candidate[:MAX_SUMMARY_PROMPT_CHARS]
            logger.info(f"Using user-supplied summary instruction ({len(candidate)} characters).")
            return candidate

        options = self.summary_options(self.prompt_policy)
        option_id = self.resolve_summary_option_id(style, options)
        option = next(o for o in options if o["id"] == option_id)
        logger.info(f"Using default summary instruction for option '{option_id}'.")
        return option["prompt"]
