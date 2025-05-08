test_case_generation:
  description: |
    Generate structured test case scenarios in Gherkin-style syntax based on the given Feature Title, Description, Acceptance Criteria, and User Journey. The scenarios must cover functional paths, edge cases, negative tests, exploratory flows, and consolidate any similar scenarios into single, comprehensive cases.
  parameters:
    - feature_title
    - feature_description
    - feature_acceptance_criteria
    - retrieved_user_journey
  prompt: |
    ## Test Case Scenario Generation Guidelines

    You are to produce a set of Gherkin-style test case scenarios that fully validate a feature.  Your scenarios must:

    - Map directly (or indirectly) to steps in the **User Journey**.  
    - Cover **positive**, **negative**, **boundary**, **edge**, **alternate**, and **exploratory** flows.  
    - **Identify and merge** any similar or overlapping scenarios into one consolidated scenario, combining their steps where sensible.  
    - Be organized, hierarchical, and richly detailed.

    ### Inputs to Review
    - **Feature Title:** `{feature_title}`  
    - **Feature Description:** `{feature_description}`  
    - **Feature Acceptance Criteria:** `{feature_acceptance_criteria}`  
    - **User Journey Information:** `{retrieved_user_journey}`

    ### Step 1: Analyze & Decompose
    1. Break down **Acceptance Criteria**, **Description**, and **User Journey** into discrete, testable behaviors.  
    2. Ensure no requirement or flow is omitted.  

    ### Step 2: Scenario Design
    For each distinct behavior or merged group of similar behaviors:
    - **Title:** Give a concise, descriptive scenario title.  
    - **Gherkin Steps:**  
      - **Given:** Precondition/context.  
      - **When:** Action(s) taken.  
      - **Then:** Expected outcome.  
      - **And:** Additional validations or steps.  

    ### Step 3: Consolidation Rule
    - If two or more scenarios differ only in minor details (e.g. copy AQM=true vs false), merge them into a single scenario with parameterized steps or multiple **When/Then/And** blocks, noting the variant conditions in the scenario title.

    ### Step 4: Output Format
    - **Strict JSON** following this template.  
    - No inner quotes in titles or scenario strings.  

    ```json
    {
      "<Test Case Scenario Title 1>": [
        {
          "scenario": "Given …\nWhen …\nThen …\nAnd …"
        }
      ],
      "<Test Case Scenario Title 2>": [
        {
          "scenario": "Given …\nWhen …\nThen …"
        }
      ]
    }
    ```

    ### Step 5: Examples

    ```json
    {
      "Verify concurrency handling in AQM tabs when engagement is carried forward or replicated": [
        {
          "scenario": "Given I am an Omnia user\nWhen I carry forward the engagement OR replicate the engagement with copy AQM set to true or false\nThen the concurrency icons and banners display if multiple users access AQM tabs\nAnd the page becomes read-only when a concurrency banner is active\nAnd a modal appears if the user navigates away without acknowledging changes"
        }
      ],

      "Validate login with valid, invalid, and blank credentials": [
        {
          "scenario": "Given the login page is displayed\nWhen I enter valid username and valid password\nThen I am redirected to the dashboard\nAnd I see a welcome message"
        },
        {
          "scenario": "Given the login page is displayed\nWhen I enter invalid username or invalid password\nThen I see an 'Invalid credentials' error\nAnd login is prevented"
        },
        {
          "scenario": "Given the login page is displayed\nWhen I leave username or password blank and click 'Login'\nThen I see a 'Required field' validation message\nAnd focus is set to the first empty field"
        }
      ],

      "Upload file size boundary and format validation": [
        {
          "scenario": "Given the file upload form is visible\nWhen I upload a .jpeg file under 5 MB\nThen the upload succeeds and displays the file in the list"
        },
        {
          "scenario": "Given the file upload form is visible\nWhen I upload a .exe or .bat file\nThen I see a 'Unsupported file type' error\nAnd the file is rejected"
        },
        {
          "scenario": "Given the file upload form is visible\nWhen I upload a PDF larger than 10 MB\nThen I see a 'File too large' error\nAnd the upload fails"
        }
      ],

      "Handle session timeout and reconnection flow": [
        {
          "scenario": "Given I am active in the app for more than 15 minutes\nWhen the session times out\nThen I see a 'Session expired' message\nAnd I am prompted to reauthenticate"
        },
        {
          "scenario": "Given my session expired\nWhen I reauthenticate successfully\nThen I return to the same page and state I was on before timeout"
        }
      ]
    }
    ```

    **Notes:**
    - **Strict Adherence:** Follow the JSON template and Gherkin patterns exactly.  
    - **Clarity:** Every step must be unambiguous and self-contained.  
    - **No Extra Commentary:** Output only the JSON structure.


Title : EAG | Smart Review | Include Gen AI Manager in filter modal in Manage Users tab


Description : 

"As a CM portal user, I want the Gen AI Manager to be displayed in the filter modal so that I can filter the users by Gen AI Manager role in the Manage Users Role.

  

**Dev Notes :**

1.  In the manage user tab , in the filter box , for role drop down add new value for GenAI Manager.
2.  Filter should start working for this new role as well."






Acceptance Criteria : 

"I know this story is completed when I see 

  

**Scenario 1: Changes in filter modal within ""Manage Country Managers"" tab**

GIVEN I have been assigned with Country Manager and/or News & Tips Manager role and/or Abandoned Files Manager role and/or Gen AI manager role

WHEN I access Country Manager portal >> ""Manage Users"" tab >> Filter options >> Select Role filter

THEN I see the newly added 4th option as ""Gen AI Manager"" in third filter

  

> Filter by
> 
> Condition
> 
> Third filter
> 
> Role
> 
> is
> 
> Dropdown:  
> Country Manager  
> News & Tips Manager  
> Abandoned Files Manager  
> Gen AI Manager
> 
>   

  

AND upon selecting the ""Gen AI Manager"" in Role filter & clicking on ""Apply"" button, all the users having that role gets filtered out and displayed in the grid 

AND I see the filter chip (filled in blue color) for the applied filter condition

AND I see all the values within the filter modal and filter chips are translated as per user preference language

  

**Testing Considerations:**

1.  Ensure to check if there is no impact on existing functionality.

-   Sorting of data.
-   Filtering of data, removing filters within filter modal, removing the applied filters through filter chip, etc.
-   Translation of data.

**Testing Considerations:**

1.  All the above ACs should be considered while testing this story.

******  
Technical Checklist******

1.  ****Architecture/Design document****
2.  ****DevOps dependencies****
3.  ****Defensive checks Implemented  
    ****
4.  ****Database update scripts****
5.  ****Unit and Integration tests****
6.  ****Local SPR verification and remediation****
7.  ****Static code scan verification and remediation****
8.  ****SonarQube Quality gate verification and remediation****
9.  ****Health check updates and remediation****
10.  ****Application insights verification and remediation****
11.  ****Monitoring and Azure Workbook updates****
12.  ****Data Validation****"


