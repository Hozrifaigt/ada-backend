import logging
import sys
import os # Import os to read environment variables

def setup_logging():
    """
    Configures the root logger for the application.
    
    Checks for an Application Insights connection string. If present, it sets up
    a handler to send logs to Azure Monitor. Otherwise, it defaults to
    logging to the console (stdout).
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set a default level

    handler = logging.StreamHandler(sys.stdout)


    # --- Check for Azure Monitor configuration ---
    # The recommended way is to use the full connection string
    # connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    # handler = None
    # if connection_string:
    #     try:
    #         # Dynamically import to avoid dependency if not used
    #         from opencensus.ext.azure.log_exporter import AzureLogHandler
            
    #         print("Azure Application Insights connection string found. Configuring AzureLogHandler.")
    #         handler = AzureLogHandler(connection_string=connection_string)
    #         # You can add custom properties to all logs sent from this instance
    #         handler.add_telemetry_processor(add_custom_properties)

    #     except ImportError:
    #         print("WARNING: opencensus-ext-azure is not installed. Falling back to console logging.")
    #         handler = logging.StreamHandler(sys.stdout)
    #     except Exception as e:
    #         print(f"ERROR: Failed to configure AzureLogHandler: {e}. Falling back to console logging.")
    #         handler = logging.StreamHandler(sys.stdout)
    # else:
    #     print("No Azure Application Insights connection string found. Configuring console logging.")
    #     handler = logging.StreamHandler(sys.stdout)

    # --- Standard Formatter (mostly for console) ---
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # --- Add the handler to the root logger ---
    if not logger.handlers:
        logger.addHandler(handler)
    else:
        # If running with uvicorn's reloader, handlers might be duplicated.
        # It's safer to clear existing handlers and add our configured one.
        logger.handlers.clear()
        logger.addHandler(handler)

# Optional: A function to add custom dimensions to all log entries
# def add_custom_properties(envelope):
#     """
#     Adds custom properties to all telemetry items.
#     'envelope' is the data object sent to Azure.
#     """
#     envelope.data.baseData.properties['service_name'] = 'AI Policy Drafter'
#     # You could add a version number, environment name, etc.
#     # envelope.data.baseData.properties['version'] = '1.0.0'
#     # envelope.data.baseData.properties['environment'] = os.getenv('ENVIRONMENT', 'development')
#     return True