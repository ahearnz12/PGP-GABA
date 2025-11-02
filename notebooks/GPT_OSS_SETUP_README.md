# FoodHub Chatbot - GPT-OSS 20B Local Setup Guide

## Overview
This notebook (`FoodHub_Chatbot_LowCode_Notebook_GPT_OSS.ipynb`) has been refactored to use the **GPT-OSS 20B model** running locally via **LM Studio** instead of OpenAI's cloud API.

## Key Changes Made

### 1. Configuration Setup
- **Removed**: OpenAI API key and cloud endpoint configuration
- **Added**: LM Studio local endpoint configuration
  - Base URL: `http://localhost:1234/v1`
  - API Key: `lm-studio` (dummy key for local use)

### 2. Model References
All instances of `gpt-4o-mini` have been replaced with `local-model` (LM Studio's default model name).

### 3. Temperature Settings
Adjusted for better performance with local models:
- **SQL Agent**: 0.1 (deterministic for accurate queries)
- **Order Query Tool**: 0.3 (focused fact extraction)
- **Answer Tool**: 0.7 (natural, friendly responses)
- **Chat Agent**: 0.5 (balanced)

### 4. New Features
- Connection test cell to verify LM Studio is running
- Clear setup instructions in the notebook header
- Error handling for connection issues

## Prerequisites

### 1. Install LM Studio
Download and install LM Studio from: https://lmstudio.ai/

### 2. Download GPT-OSS 20B Model
1. Open LM Studio
2. Go to the "Discover" or "Models" tab
3. Search for "GPT-OSS 20B" or the specific model variant
4. Click "Download" and wait for completion

### 3. Start Local Server
1. In LM Studio, go to the "Local Server" tab
2. Select the GPT-OSS 20B model from the dropdown
3. Click "Start Server"
4. Verify it's running on `http://localhost:1234`

## Running the Notebook

### Step-by-Step Execution

1. **Start LM Studio Server** (before running any cells!)
   - Load GPT-OSS 20B model
   - Start the local server
   - Verify server is running

2. **Run Cell 1**: Install dependencies
   ```bash
   pip install openai langchain langchain-openai ...
   ```

3. **Run Cell 2**: Import libraries

4. **Run Cell 3**: Configure LM Studio endpoint
   - Should print: "✓ LM Studio configuration set"

5. **Run Cell 4**: Test connection
   - Should print: "✓ LM Studio Connection Successful!"
   - If fails, check LM Studio is running

6. **Continue with remaining cells** as normal

## Troubleshooting

### "Connection Failed" Error
**Problem**: Cannot connect to LM Studio
**Solutions**:
- Verify LM Studio is running
- Check local server is started in LM Studio
- Ensure port 1234 is not blocked
- Try restarting LM Studio

### Slow Response Times
**Problem**: Model takes too long to respond
**Solutions**:
- GPT-OSS 20B requires significant compute (16GB+ RAM recommended)
- Close other resource-intensive applications
- Consider using a smaller model if hardware is limited
- Enable GPU acceleration in LM Studio if available

### "Model Not Found" Error
**Problem**: LM Studio can't find the model
**Solutions**:
- Verify model is fully downloaded
- Check model name matches "local-model" or update in code
- Restart LM Studio and reload the model

### Poor Quality Responses
**Problem**: Responses are nonsensical or off-topic
**Solutions**:
- Verify you're using GPT-OSS 20B (not a smaller model)
- Check temperature settings aren't too high
- Ensure prompts are clear and well-structured
- Try reloading the model in LM Studio

## Performance Considerations

### Hardware Requirements
- **Minimum**: 16GB RAM, 8-core CPU
- **Recommended**: 32GB RAM, 16-core CPU, GPU with 8GB+ VRAM
- **Storage**: ~15GB for GPT-OSS 20B model

### Response Times
- **Expected**: 2-10 seconds per query (depending on hardware)
- **Factors**: Prompt length, context size, hardware specs

### Optimization Tips
1. Enable GPU acceleration in LM Studio settings
2. Adjust context length in LM Studio (shorter = faster)
3. Use batch processing for multiple queries
4. Close unnecessary applications to free RAM

## Differences from Cloud Version

| Aspect | Cloud (OpenAI) | Local (LM Studio) |
|--------|---------------|-------------------|
| **Cost** | Pay per token | Free (after setup) |
| **Privacy** | Data sent to cloud | Completely local |
| **Speed** | Fast (1-2 sec) | Moderate (2-10 sec) |
| **Setup** | API key only | Model download + server |
| **Hardware** | None required | 16GB+ RAM needed |
| **Availability** | Internet required | Works offline |

## Benefits of Local Deployment

1. **Privacy**: Customer data never leaves your machine
2. **Cost**: No per-token charges after initial setup
3. **Offline**: Works without internet connection
4. **Control**: Full control over model and parameters
5. **Customization**: Can fine-tune or swap models easily

## Switching Back to Cloud Version

If you need to use the cloud version:
1. Use the original notebook: `FoodHub_Chatbot_LowCode_Notebook.ipynb`
2. Or update the configuration cells to use OpenAI endpoints:
   ```python
   os.environ['OPENAI_API_KEY'] = "your-api-key"
   os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
   model_name = "gpt-4o-mini"
   ```

## Support

For issues specific to:
- **LM Studio**: Visit https://lmstudio.ai/docs
- **GPT-OSS Model**: Check model documentation
- **Notebook Code**: Review cell outputs and error messages

## Version Information

- **Notebook Version**: 1.0 (GPT-OSS)
- **LangChain Version**: 0.3.26
- **OpenAI Python Version**: 1.93.0 (compatible with local endpoints)
- **LM Studio**: Latest version from lmstudio.ai

---

**Last Updated**: October 2025
