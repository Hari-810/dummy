import json
import logging 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from data_generator.base_generator import BaseSyntheticGenerator
from config.constants import DATA_CATEGORY, GENERIC_VARIABLES, NT_VARIABLES

class TextualDataGenerator(BaseSyntheticGenerator):
    def __init__(self):
        """
        Initializes the TextualDataGenerator class by calling the base class constructor.
        Sets up the logger for this class.
        """
        super().__init__(data_category=DATA_CATEGORY.TEXTUAL.value)
        self.logger = logging.getLogger(__name__)

    def json_to_plain_text(self,data, indent=0):
        """
        Converts a JSON-like structure (dict or list) to a human-readable plain text format.

        Args:
            data (dict or list): The JSON-like data to be converted into plain text.
            indent (int): The number of spaces to use for indentation at each level of nesting. Defaults to 0.

        Returns:
            str: A plain-text representation of the input data, with proper indentation.
        """
        lines = []
        if isinstance(data, dict):
            for key, value in data.items():
                prefix = "  " * indent + f"{key.replace('_', ' ').capitalize()}: "
                if isinstance(value, (dict, list)):
                    lines.append(prefix)
                    lines.append(self.json_to_plain_text(value, indent + 1))
                else:
                    lines.append(prefix + str(value))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append("  " * indent + f"- {self.json_to_plain_text(item, indent + 1)}")
                else:
                    lines.append("  " * indent + f"- {item}")
        else:
            lines.append("  " * indent + str(data))
        return "\n".join(lines)

    def generate_synthetic_data(self,data_category:str, user_query: str, num_samples: int,realism_level=str,text_style=str,
                                text_structure=str, domain=str, logical_constraints=str,output_format=str):
        """
        Generates synthetic data in batches using Azure OpenAI LLM.
        Parameters:
            user_query (str): User's query describing the data to generate.
            num_samples (int): Total number of synthetic sentences to generate.
            realism_level (str): The realism level for data generation.
            text_style(str): The style of the text to be generated.
            text_structure(str): The structure of the text to be generated
            domain (str): The domain for the synthetic data.
            logical_constraints (str): The logical constraints for the data.
            output_format (dict): The format of the output file (CSV, JSON,TXT).
        Returns:
            list: The generated synthetic data.
        """
        generated_texts = []
        parser = JsonOutputParser()
       
        self.prompt_template = PromptTemplate(
            template=self.prompt_template[NT_VARIABLES.YAML_FILE_NAME],
            input_variables=[NT_VARIABLES.USER_REQUEST,NT_VARIABLES.DATA_CATEGORY, GENERIC_VARIABLES.SAMPLE_COUNT,
                             GENERIC_VARIABLES.REALISM_LEVEL,GENERIC_VARIABLES.DOMAIN, NT_VARIABLES.TEXT_STYLE,
                             NT_VARIABLES.TEXT_STRUCTURE,NT_VARIABLES.LOGICAL_CONSTRAINTS, GENERIC_VARIABLES.OUTPUT_FORMAT],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        total_generated = 0

        formatted_prompt = self.prompt_template.template.format(
        data_category = str(data_category),
        user_request=str(user_query),
        num_samples=int(num_samples),
        realism_level=str(realism_level).strip(), 
        domain=str(domain),
        text_style=str(text_style),
        text_structure=str(text_structure),
        logical_constraints=str(logical_constraints),
        output_format=str(output_format),
        )
                    
        try:
            response = self.llm_model.invoke([HumanMessage(content=formatted_prompt)])
            logging.info(f"Raw LLM response : {response}")
            if response and hasattr(response, "content"):
                logging.info(f"The LLM Response is:{response.content}")
                cleaned_response = response.content.strip().strip("```json").strip("```")
                parsed_json = json.loads(cleaned_response)
                text_data_response = self.json_to_plain_text(parsed_json)
                generated_texts.append(text_data_response )
                logging.info(f"the cleaned response is:{text_data_response}" )
                total_generated += len(generated_texts)
        except Exception as e:
            logging.error(f"Error generating synthetic data: {e}")

        if not generated_texts:
            logging.error(f"No synthetic data was generated.")
            return None   

        return generated_texts,total_generated
    
    def save_generated_data(self, generated_texts, base_filename: str, output_format: str):
        """
        Save generated textual data in the specified format using the base class logic.
        
        Args:
            generated_texts (list): The list of generated text data to be saved.
            base_filename (str): The base name for the saved file (used for naming the file).
            output_format (str): The format in which to save the data (e.g., "txt", "json").
            
        Returns:
            None
        """
        if isinstance(generated_texts, list) and generated_texts:
            # Save the generated textual data using the base class logic
            self.save_output(generated_texts, base_filename, output_format)
            logging.info("Synthetic Textual data saved successfully.")
        else:
            logging.error("No valid text data to save.")



            

    

 
    
  
       