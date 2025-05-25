import logging
from .tabular_data_generator import TabularDataGenerator
from .textual_data_generator import TextualDataGenerator
from config.constants import GENERIC_VARIABLES, TDC_VARIABLES, NT_VARIABLES, DATA_CATEGORY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SyntheticDataGeneration:
    """
    Handles orchestration of synthetic data generation for different data categories 
    (e.g., tabular, natural text) using appropriate generator classes.
    """

    def __init__(self):
        self.generator = None
        self.synthetic_data = None
        self.base_filename = None
        self.total_generated = 0

    def generate_tabular_data(self, user_inputs: dict):
        """
        Triggers tabular data generation using the TabularDataGenerator.

        Args:
            user_inputs (dict): Input configuration and parameters.

        Returns:
            pd.DataFrame: Generated tabular data.
        """
        logger.debug("Initializing TabularDataGenerator with user inputs.")
        self.generator = TabularDataGenerator()
        return self.generator.generate_synthetic_data(
            data_category=DATA_CATEGORY.TABULAR.value,
            user_query=user_inputs.get(TDC_VARIABLES.USER_REQUEST),
            num_samples=user_inputs.get(GENERIC_VARIABLES.SAMPLE_COUNT),
            realism_level=user_inputs.get(GENERIC_VARIABLES.REALISM_LEVEL, GENERIC_VARIABLES.DEFAULT_REALISM_LEVEL),
            column_names=list(user_inputs.get(TDC_VARIABLES.SCHEMA_DETAILS, {}).keys()),
            field_description=user_inputs.get(TDC_VARIABLES.SCHEMA_DETAILS),
            max_length=user_inputs.get(TDC_VARIABLES.SCHEMA_DETAILS),
            domain=user_inputs.get(GENERIC_VARIABLES.DOMAIN, GENERIC_VARIABLES.DEFAULT_DOMAIN),
            feature_types=user_inputs.get(TDC_VARIABLES.FEATURE_TYPES, TDC_VARIABLES.DEFAULT_FEATURE_TYPES),
            feature_distributions=user_inputs.get(TDC_VARIABLES.FEATURE_DISTRIBUTION, TDC_VARIABLES.DEFAULT_FEATURE_DISTRIBUTION),
            relationships_constraints=user_inputs.get(TDC_VARIABLES.RELATIONSHIP_CONSTRAINTS, ''),
            output_format=user_inputs.get(GENERIC_VARIABLES.OUTPUT_FORMAT, TDC_VARIABLES.DEFAULT_OUTPUT_FORMAT),
            schema_details=user_inputs.get(TDC_VARIABLES.SCHEMA_DETAILS, {})
        )

    def generate_textual_data(self, user_inputs: dict):
            """
            Triggers natural text data generation using the TextualDataGenerator.

            Args:
                user_inputs (dict): Input configuration and parameters.

            Returns:
                List[str] or str: Generated natural language text.
            """
            logger.debug("Initializing TextualDataGenerator with user inputs.")
            self.generator = TextualDataGenerator()
            return self.generator.generate_synthetic_data(
                data_category=DATA_CATEGORY.TEXTUAL.value,
                user_query=user_inputs.get(GENERIC_VARIABLES.USER_QUERY, ''),
                num_samples=user_inputs.get(GENERIC_VARIABLES.SAMPLE_COUNT, NT_VARIABLES.DEFAULT_SAMPLE_COUNT),
                realism_level=user_inputs.get(GENERIC_VARIABLES.REALISM_LEVEL, GENERIC_VARIABLES.DEFAULT_REALISM_LEVEL),
                text_style=user_inputs.get(NT_VARIABLES.TEXT_STYLE, NT_VARIABLES.DEFAULT_TEXT_STYLE),
                text_structure=user_inputs.get(NT_VARIABLES.TEXT_STRUCTURE, NT_VARIABLES.DEFAULT_TEXT_STRUCTURE),
                domain=user_inputs.get(GENERIC_VARIABLES.DOMAIN, GENERIC_VARIABLES.DEFAULT_DOMAIN),
                logical_constraints=user_inputs.get(NT_VARIABLES.LOGICAL_CONSTRAINTS, ''),
                output_format=user_inputs.get(GENERIC_VARIABLES.OUTPUT_FORMAT, NT_VARIABLES.DEFAULT_OUTPUT_FORMAT)
            )

    def main(self, data_category: str, user_inputs: dict):
        """
        Entry point for generating and saving synthetic data.

        Args:
            data_category (str): Either "Tabular" or "Natural Text".
            user_inputs (dict): Dictionary of all relevant parameters and inputs.

        Returns:
            Any: The generated synthetic data or None if failed.
        """
        try:
            self.base_filename = data_category.lower()
            logger.info(f"Initiating generation for category: {data_category}")

            if data_category.lower() == DATA_CATEGORY.TABULAR.value:
                logger.info("Using TabularDataGenerator.")
                self.synthetic_data,self.total_generated = self.generate_tabular_data(user_inputs)

            elif data_category.lower() == DATA_CATEGORY.TEXTUAL.value:
                logger.info("Using TextualDataGenerator.")
                self.synthetic_data,self.total_generated = self.generate_textual_data(user_inputs)
              
            if self.generator and self.synthetic_data is not None:
                output_format = user_inputs.get(GENERIC_VARIABLES.OUTPUT_FORMAT, "json").lower()
                filename = f"{self.base_filename}_data.{output_format}"
                logger.info(f"Saving synthetic data as: {filename}")
                self.generator.save_generated_data(self.synthetic_data,base_filename=self.base_filename, output_format=output_format)

            logger.info("Data generation completed successfully.")
            return self.synthetic_data,self.total_generated

        except Exception as e:
            logger.exception("Unexpected error during synthetic data generation")
            return None,0