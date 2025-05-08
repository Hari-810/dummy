class TestCaseQueryPrompter(Prompter):
    """
    Builds a structured prompt to generate Gherkin-style test case scenarios based on feature inputs and user journey.
    """

    def generate_prompt(self, step_context, **inputs) -> str:
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
- **Feature Title:** {inputs.get("feature_title", "EAG | Smart Review | Include Gen AI Manager in filter modal in Manage Users tab")}  
- **Feature Description:** {inputs.get("feature_description", "As a CM portal user, I want the Gen AI Manager to be displayed in the filter modal so that I can filter the users by Gen AI Manager role in the Manage Users Role.")}  
- **Feature Acceptance Criteria:** {inputs.get("feature_acceptance_criteria", "Scenario 1: Changes in filter modal within \"Manage Country Managers\" tab\n\nGIVEN I have been assigned with Country Manager and/or News & Tips Manager role and/or Abandoned Files Manager role and/or Gen AI manager role\n\nWHEN I access Country Manager portal >> \"Manage Users\" tab >> Filter options >> Select Role filter\n\nTHEN I see the newly added 4th option as \"Gen AI Manager\" in third filter\n\nAND upon selecting the \"Gen AI Manager\" in Role filter & clicking on \"Apply\" button, all the users having that role gets filtered out and displayed in the grid\n\nAND I see the filter chip (filled in blue color) for the applied filter condition\n\nAND I see all the values within the filter modal and filter chips are translated as per user preference language\n\nTesting Considerations:\n- Sorting of data.\n- Filtering of data, removing filters within filter modal, removing the applied filters through filter chip, etc.\n- Translation of data.")}  
- **User Journey Information:** {inputs.get("retrieved_user_journey", "User opens CM Portal > Navigates to Manage Users tab > Opens filter modal > Selects Role filter > Selects 'Gen AI Manager' > Applies filter > Views filtered user list with blue chip visible > Changes user language > Verifies translation.")}

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
"""

    def _get_notes_section(self) -> str:
        return """**Notes:**"""
