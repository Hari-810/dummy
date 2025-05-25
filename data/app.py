import streamlit as st
from frontend.ui.sidebar import Sidebar
from frontend.ui.schema_editor import SchemaEditor
from frontend.ui.display_results import DisplayResults
from frontend.utils.payload_builder import DataPayloadBuilder
from data_generator.main import SyntheticDataGeneration

class SyntheticDataApp:
    """
    A Streamlit-based application for generating synthetic data based on user-defined configurations.
    The app allows users to select data categories, define schemas, and generate synthetic data.
    It integrates with various modules for schema definition, data generation, and result display.
    """

    def __init__(self):
        """
        Initializes the SyntheticDataApp by setting up necessary components:
        - Data generation engine
        - Sidebar for configuration
        - Schema editor for tabular data
        - Display results handler
        - Payload builder to create requests
        """
        self.generator = SyntheticDataGeneration()
        self.sidebar = Sidebar()
        self.schema_editor = SchemaEditor()
        self.display_results = DisplayResults()
        self.payload_builder = DataPayloadBuilder()

    def trigger_generation(self, config, structured_schema):
        try:
            payload = self.payload_builder.build_payload(config, structured_schema)
            synthetic_data, total_generated = self.generator.main(config["data_category"], payload)
            self.display_results.render_results(synthetic_data, config, total_generated)
        except Exception as e:
            st.error(f"Error: {e}")

    def run(self):
        """
        Runs the Streamlit app, initializing the page configuration, displaying the app title,
        rendering the sidebar, schema editor, and handling data generation. 

        This method collects user inputs, validates configurations, and triggers the data generation process.
        After generation, it displays the results in the appropriate format.
        """
        # Set up Streamlit page configuration
        st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
        st.title("Synthetic Data Generator")
        st.write("Generate high-quality synthetic data tailored to your domain and data characteristics.")

        # Sidebar input configuration
        sidebar_config = self.sidebar.render_sidebar()

        # Schema editor for Tabular data only
        structured_schema, field_description,is_valid_schema = self.schema_editor.render_schema_editor(sidebar_config)
        if sidebar_config["data_category"] == "Tabular":
            sidebar_config["schema_details"] = structured_schema
            sidebar_config["field_description"] = field_description
        # Schema Preview
        if structured_schema:
            st.divider()
            st.subheader("Schema preview")
            st.caption("ℹ️ Click to preview the schema of the data")
            with st.expander("schema preview", expanded=True):
                self.schema_editor.show_schema_preview(structured_schema)

        # Generate Data section
        st.divider()
        st.subheader("Generate Data")
        if st.button("Start Generation", use_container_width=True, disabled=not sidebar_config["is_supported"]):
            if sidebar_config["data_category"] == "Tabular":
                if not structured_schema:
                    st.error("Please define at least one column in the schema.")
                elif not is_valid_schema:
                    st.error("Please complete all required schema fields: Column Name, Data Type, and Description.")
                else:
                    with st.spinner("Generating Synthetic Data..."):
                        self.trigger_generation(sidebar_config, structured_schema)
            else:
                self.trigger_generation(sidebar_config, structured_schema)

if __name__ == '__main__':
    app = SyntheticDataApp()
    app.run()