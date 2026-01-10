# Deployment Guide: Options Analytics Dashboard

This guide will walk you through deploying the Options Analytics Dashboard to Streamlit Cloud, making it publicly accessible to anyone.

## Prerequisites

- A GitHub account
- Your code pushed to a GitHub repository
- A Streamlit Cloud account (free)

## Step-by-Step Deployment Instructions

### Step 1: Prepare Your GitHub Repository

1. **Initialize Git** (if not already done):
   ```bash
   cd options_project
   git init
   ```

2. **Create a .gitignore file** (if not exists) to exclude unnecessary files:
   ```
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .Python
   .pytest_cache/
   .mypy_cache/
   *.egg-info/
   dist/
   build/
   .env
   .venv
   venv/
   ```

3. **Commit and push your code**:
   ```bash
   git add .
   git commit -m "Initial commit: Options Analytics Dashboard"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### Step 2: Sign Up for Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** and authenticate with your GitHub account
3. Authorize Streamlit Cloud to access your GitHub repositories

### Step 3: Deploy Your App

1. **Click "New app"** button in the Streamlit Cloud dashboard

2. **Select your repository**:
   - Choose the GitHub repository containing your options project
   - Select the branch (usually `main` or `master`)

3. **Configure the app**:
   - **Main file path**: Enter `dashboard/app.py`
     - Alternatively, if you use `streamlit_app.py` at the root, leave it as `streamlit_app.py`
   - **App URL**: Choose a unique name for your app (e.g., `options-analytics`)
     - Your app will be available at: `https://options-analytics.streamlit.app`

4. **Advanced settings** (optional):
   - Python version: 3.9 or later (default is usually fine)
   - Secrets: Add any environment variables if needed (not required for this app)

5. **Click "Deploy"**

### Step 4: Wait for Deployment

- Streamlit Cloud will automatically:
  - Install dependencies from `requirements.txt`
  - Build your app
  - Deploy it to a public URL
- This usually takes 1-3 minutes
- You'll see build logs in real-time

### Step 5: Access Your Deployed App

- Once deployment completes, click **"Manage app"** or visit your app URL
- Your app is now publicly accessible!
- Share the URL with anyone: `https://your-app-name.streamlit.app`

## Automatic Updates

Streamlit Cloud automatically redeploys your app whenever you push changes to your GitHub repository. No manual action needed!

## Troubleshooting

### Build Fails

**Issue**: Build fails with import errors
- **Solution**: Ensure all dependencies are in `requirements.txt`
- Check that the main file path is correct (`dashboard/app.py`)

**Issue**: Module not found errors
- **Solution**: Verify your project structure is correct
- Ensure all `__init__.py` files exist in package directories
- Check that imports use relative paths correctly

### App Runs But Shows Errors

**Issue**: Import errors in the app
- **Solution**: The app uses `sys.path.insert()` to add parent directory
- This should work, but if issues persist, verify the file structure matches the code

**Issue**: Missing dependencies
- **Solution**: Add any missing packages to `requirements.txt`
- Common additions might include: `scipy`, `numpy`, `pandas`, `matplotlib`, `plotly`

### Performance Issues

**Issue**: App is slow or times out
- **Solution**: Streamlit Cloud free tier has resource limits
- Optimize Monte Carlo simulations (reduce default number of simulations)
- Consider caching expensive computations with `@st.cache_data`

### Configuration Issues

**Issue**: Theme or settings not applied
- **Solution**: Ensure `.streamlit/config.toml` is committed to your repository
- Streamlit Cloud reads configuration from this file

## Alternative Deployment Options

If Streamlit Cloud doesn't meet your needs, consider:

1. **Render** (https://render.com)
   - Free tier available
   - Requires `Procfile` or `render.yaml`

2. **Railway** (https://railway.app)
   - Free tier available
   - Simple deployment process

3. **Heroku** (https://www.heroku.com)
   - Paid service
   - Requires `Procfile` and `runtime.txt`

4. **AWS/GCP/Azure**
   - More complex setup
   - Requires Docker or container configuration

## Security Considerations

- Your app is **publicly accessible** - anyone with the URL can use it
- No authentication is configured by default
- Consider adding authentication if you need to restrict access:
  - Use Streamlit's built-in authentication (Streamlit Cloud Pro)
  - Or implement custom authentication in the app

## Cost

- **Streamlit Cloud**: Free for public apps
- **Streamlit Cloud Pro**: Paid for private apps and additional features

## Support

- Streamlit Cloud documentation: https://docs.streamlit.io/streamlit-cloud
- Streamlit Community: https://discuss.streamlit.io

---

**Your app is now live! 🚀**

Share your deployment URL: `https://your-app-name.streamlit.app`
