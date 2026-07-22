"""
Configuration loader for Lyr.

Loads environment variables from .env file and provides
configuration values to the rest of the application.

Environment variables:
    LYR_API_KEY: Required. API key for authenticating requests.
    LYR_ENVIRONMENT: Optional. 'development' or 'production'. Default: 'development'.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root (where .env should be)
# This goes up from config.py: recognition/config.py -> recognition/ -> project root
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env file if it exists
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Try current directory as fallback
    load_dotenv()


class Config:
    """Application configuration."""
    
    # API Key for authentication
    API_KEY: str = os.getenv("LYR_API_KEY", "")
    
    # Environment: development or production
    ENVIRONMENT: str = os.getenv("LYR_ENVIRONMENT", "development")
    
    # Host and port
    HOST: str = os.getenv("LYR_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("LYR_PORT", "8000"))
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate required configuration.
        
        Raises:
            ValueError: If required config is missing.
        """
        if not cls.API_KEY:
            raise ValueError(
                "LYR_API_KEY environment variable is not set.\n"
                f"Create a .env file in {PROJECT_ROOT} with:\n"
                "LYR_API_KEY=your_secure_api_key_here\n"
            )
        
        if len(cls.API_KEY) < 16:
            raise ValueError(
                "LYR_API_KEY is too short. Use at least 16 characters.\n"
                f"Current length: {len(cls.API_KEY)}"
            )
    
    @classmethod
    def is_development(cls) -> bool:
        """Return True if running in development mode."""
        return cls.ENVIRONMENT.lower() == "development"
    
    @classmethod
    def is_production(cls) -> bool:
        """Return True if running in production mode."""
        return cls.ENVIRONMENT.lower() == "production"


# Validate config on import
Config.validate()