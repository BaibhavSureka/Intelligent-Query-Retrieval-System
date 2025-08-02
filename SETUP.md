# Quick Setup Guide

## Prerequisites
- Python 3.10 or higher
- OpenAI API key

## Quick Start (Local)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Run the application**:
   ```bash
   python run.py
   ```

4. **Test the API**:
   ```bash
   python test_api.py
   ```

## Quick Deploy

### Railway
1. Connect your GitHub repo to Railway
2. Set `OPENAI_API_KEY` environment variable
3. Deploy automatically

### Render
1. Connect your GitHub repo to Render
2. Set `OPENAI_API_KEY` in dashboard
3. Deploy automatically

### Heroku
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY="your-key"
git push heroku main
```

## API Usage

**Main Endpoint**: `POST /hackrx/run`

**Headers**: `Authorization: Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b`

**Sample Request**:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is covered?", "What are the conditions?"]
}
```

**Sample Response**:
```json
{
  "answers": ["Answer 1", "Answer 2"]
}
```

## Need Help?
- Check `README.md` for detailed documentation
- Run `python test_api.py` to verify setup
- Check logs for error details