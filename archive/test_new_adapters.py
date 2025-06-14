import asyncio
import os
import unittest
from unittest.mock import patch, AsyncMock
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import adapters
try:
    from src.mcts_mcp_server.openai_adapter import OpenAIAdapter
except ImportError:
    OpenAIAdapter = None
    logger.warning("Could not import OpenAIAdapter, tests for it will be skipped.")

try:
    from src.mcts_mcp_server.anthropic_adapter import AnthropicAdapter
except ImportError:
    AnthropicAdapter = None
    logger.warning("Could not import AnthropicAdapter, tests for it will be skipped.")

try:
    from src.mcts_mcp_server.gemini_adapter import GeminiAdapter
except ImportError:
    GeminiAdapter = None
    logger.warning("Could not import GeminiAdapter, tests for it will be skipped.")

# Helper function to run async tests
def async_test(f):
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

class TestNewAdapters(unittest.TestCase):

    @unittest.skipIf(OpenAIAdapter is None, "OpenAIAdapter not imported")
    @patch.dict(os.environ, {}, clear=True) # Start with a clean environment
    def test_openai_adapter_no_key(self):
        if OpenAIAdapter is None:
            self.skipTest("OpenAIAdapter not available")
        logger.info("Testing OpenAIAdapter initialization without API key...")
        with self.assertRaisesRegex(ValueError, "OpenAI API key not provided"):
            OpenAIAdapter()

    @unittest.skipIf(OpenAIAdapter is None, "OpenAIAdapter not imported")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("openai.AsyncOpenAI") # Mock the actual client
    @async_test
    async def test_openai_adapter_with_key_mocked_completion(self, MockAsyncOpenAI):
        if OpenAIAdapter is None:
            self.skipTest("OpenAIAdapter not available")
        logger.info("Testing OpenAIAdapter with key and mocked completion...")
        # Configure the mock client and its methods
        mock_client_instance = MockAsyncOpenAI.return_value
        mock_completion_response = AsyncMock()
        mock_completion_response.choices = [AsyncMock(message=AsyncMock(content="Mocked OpenAI response"))]
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion_response)

        adapter = OpenAIAdapter(api_key="test_key")
        self.assertIsNotNone(adapter.client)
        response = await adapter.get_completion(model=None, messages=[{"role": "user", "content": "Hello"}])
        self.assertEqual(response, "Mocked OpenAI response")
        MockAsyncOpenAI.assert_called_with(api_key="test_key")
        mock_client_instance.chat.completions.create.assert_called_once()

    @unittest.skipIf(AnthropicAdapter is None, "AnthropicAdapter not imported")
    @patch.dict(os.environ, {}, clear=True)
    def test_anthropic_adapter_no_key(self):
        if AnthropicAdapter is None:
            self.skipTest("AnthropicAdapter not available")
        logger.info("Testing AnthropicAdapter initialization without API key...")
        with self.assertRaisesRegex(ValueError, "Anthropic API key not provided"):
            AnthropicAdapter()

    @unittest.skipIf(AnthropicAdapter is None, "AnthropicAdapter not imported")
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True)
    @patch("anthropic.AsyncAnthropic") # Mock the actual client
    @async_test
    async def test_anthropic_adapter_with_key_mocked_completion(self, MockAsyncAnthropic):
        if AnthropicAdapter is None:
            self.skipTest("AnthropicAdapter not available")
        logger.info("Testing AnthropicAdapter with key and mocked completion...")
        mock_client_instance = MockAsyncAnthropic.return_value
        mock_completion_response = AsyncMock()
        # Anthropic's response structure for content is a list of blocks
        mock_response_content_block = AsyncMock()
        mock_response_content_block.text = "Mocked Anthropic response"
        mock_completion_response.content = [mock_response_content_block]

        mock_client_instance.messages.create = AsyncMock(return_value=mock_completion_response)

        adapter = AnthropicAdapter(api_key="test_key")
        self.assertIsNotNone(adapter.client)
        # Provide a simple messages list that _prepare_anthropic_messages_and_system_prompt can handle
        response = await adapter.get_completion(model=None, messages=[{"role": "user", "content": "Hello"}])
        self.assertEqual(response, "Mocked Anthropic response")
        MockAsyncAnthropic.assert_called_with(api_key="test_key")
        mock_client_instance.messages.create.assert_called_once()

    @unittest.skipIf(GeminiAdapter is None, "GeminiAdapter not imported")
    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_adapter_no_key(self):
        if GeminiAdapter is None:
            self.skipTest("GeminiAdapter not available")
        logger.info("Testing GeminiAdapter initialization without API key...")
        with self.assertRaisesRegex(ValueError, "Gemini API key not provided"):
            GeminiAdapter()

    @unittest.skipIf(GeminiAdapter is None, "GeminiAdapter not imported")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("google.generativeai.GenerativeModel") # Mock the actual client
    @patch("google.generativeai.configure") # Mock configure
    @async_test
    async def test_gemini_adapter_with_key_mocked_completion(self, mock_genai_configure, MockGenerativeModel):
        if GeminiAdapter is None:
            self.skipTest("GeminiAdapter not available")
        logger.info("Testing GeminiAdapter with key and mocked completion...")
        mock_model_instance = MockGenerativeModel.return_value
        # Ensure the mock response object has a 'text' attribute directly if that's what's accessed
        mock_generate_content_response = AsyncMock()
        mock_generate_content_response.text = "Mocked Gemini response"
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_generate_content_response)

        adapter = GeminiAdapter(api_key="test_key")
        self.assertIsNotNone(adapter.client)
        # Provide a simple messages list that _convert_messages_to_gemini_format can handle
        response = await adapter.get_completion(model=None, messages=[{"role": "user", "content": "Hello"}])
        self.assertEqual(response, "Mocked Gemini response")
        mock_genai_configure.assert_called_with(api_key="test_key")
        # Check if GenerativeModel was called with the default model name from the adapter
        MockGenerativeModel.assert_called_with(adapter.model_name)
        mock_model_instance.generate_content_async.assert_called_once()

if __name__ == "__main__":
    unittest.main()
