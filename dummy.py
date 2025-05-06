from graph_of_thoughts.prompter import Prompter

class TestCaseQueryPrompter(Prompter):
    """
    Builds a structured prompt to generate Gherkin-style test case scenarios based on feature inputs and user journey.
    """

    def generate_prompt(self, inputs: dict) -> str:
        return (
            self._get_header()
            + self._get_inputs_section(inputs)
            + self._get_analysis_section()
            + self._get_scenario_design_section()
            + self._get_consolidation_section()
            + self._get_output_format_section()
            + self._get_notes_section()
        )

    def aggregation_prompt(self, inputs: dict) -> str:
        return "Combine similar test case scenarios into one, using parameterized steps where necessary."

    def improve_prompt(self, inputs: dict) -> str:
        return "Review and enhance scenario coverage for edge cases, boundary conditions, and missed flows."

    def score_prompt(self, inputs: dict) -> str:
        return "Assign a score from 1-10 based on completeness, clarity, and Gherkin compliance of the test cases."

    def validation_prompt(self, inputs: dict) -> str:
        return "Ensure that all Acceptance Criteria and User Journey steps are covered by at least one scenario."

    def _get_header(self) -> str:
        return """## Test Case Scenario Generation Guidelines

You are to produce a set of Gherkin-style test case scenarios that fully validate a feature. Your scenarios must:

- Map directly (or indirectly) to steps in the **User Journey**.  
- Cover **positive**, **negative**, **boundary**, **edge**, **alternate**, and **exploratory** flows.  
- **Identify and merge** any similar or overlapping scenarios into one consolidated scenario.  
- Be organized, hierarchical, and richly detailed.

"""

    def _get_inputs_section(self, inputs: dict) -> str:
        return f"""### Inputs to Review
- **Feature Title:** {inputs.get("feature_title", "")}  
- **Feature Description:** {inputs.get("feature_description", "")}  
- **Feature Acceptance Criteria:** {inputs.get("feature_acceptance_criteria", "")}  
- **User Journey Information:** {inputs.get("retrieved_user_journey", "")}

"""

    def _get_analysis_section(self) -> str:
        return """### Step 1: Analyze & Decompose
1. Break down **Acceptance Criteria**, **Description**, and **User Journey** into discrete, testable behaviors.  
2. Ensure no requirement or flow is omitted.  

"""

    def _get_scenario_design_section(self) -> str:
        return """### Step 2: Scenario Design
For each distinct behavior or merged group of similar behaviors:
- **Title:** Give a concise, descriptive scenario title.  
- **Gherkin Steps:**  
  - **Given:** Precondition/context.  
  - **When:** Action(s) taken.  
  - **Then:** Expected outcome.  
  - **And:** Additional validations or steps.  

"""

    def _get_consolidation_section(self) -> str:
        return """### Step 3: Consolidation Rule
- If scenarios differ only in minor details (e.g. copy AQM = true vs false), merge into one with multiple **When/Then/And** blocks or parameterized logic.

"""

    def _get_output_format_section(self) -> str:
        return """### Step 4: Output Format
- **Strict JSON** format without extra quotes or markdown.  

```json
{
  "<Test Case Scenario Title>": [
    {
      "scenario": "Given ...\\nWhen ...\\nThen ...\\nAnd ..."
    }
  ]
}


    def _get_notes_section(self) -> str:
    return """**Notes:**"""
