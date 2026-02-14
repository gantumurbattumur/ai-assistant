"""Configuration and environment setup"""
import getpass
import os


def set_env(key: str):
    """Set environment variable if not already set"""
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


def setup_environment():
    """Setup required environment variables"""
    set_env("OPENAI_API_KEY")
    set_env("TAVILY_API_KEY")
