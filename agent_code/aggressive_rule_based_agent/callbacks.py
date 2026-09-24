"""Rule-based opponent that prioritizes pursuing other agents."""

from agent_code.rule_based_agent.callbacks import act, setup as base_setup


def setup(self):
    base_setup(self)
    self.prioritize_opponents = True
