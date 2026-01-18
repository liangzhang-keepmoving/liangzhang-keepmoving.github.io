# Deployment Guide for Liang Zhang's Personal Website

## Overview
This guide will help you deploy your personal website to GitHub Pages.

## Prerequisites
- A GitHub account (already created: liangzhang-keepmoving)
- Git installed on your local machine
- Your personal website files (already prepared)

## Step 1: Create a GitHub Repository
1. Log in to your GitHub account at https://github.com
2. Click on the "+" icon in the top right corner and select "New repository"
3. Repository name: `liangzhang-keepmoving.github.io` (this is required for GitHub Pages)
4. Description: "Personal website for Liang Zhang"
5. Select "Public" repository
6. Do NOT initialize with README, .gitignore, or license (we already have these files)
7. Click "Create repository"

## Step 2: Configure Git and Push Code
1. Open Terminal and navigate to your website directory:
   ```bash
   cd /Users/liangzhang/Desktop/code/LS/skills/personal_website
   ```

2. Make sure your Git configuration is correct:
   ```bash
   git config user.name "liangzhang-keepmoving"
   git config user.email "liya.liang.zhang@gmail.com"
   ```

3. Add the remote repository URL:
   ```bash
   git remote set-url origin https://github.com/liangzhang-keepmoving/liangzhang-keepmoving.github.io.git
   ```

4. Push your code to GitHub:
   ```bash
   git push -u origin main
   ```

5. Enter your GitHub credentials when prompted (or use SSH keys for authentication)

## Step 3: Enable GitHub Pages
1. Go to your repository settings at https://github.com/liangzhang-keepmoving/liangzhang-keepmoving.github.io/settings
2. Scroll down to the "GitHub Pages" section
3. Under "Source", select "Branch: main" and click "Save"
4. Wait a few minutes for GitHub to build and deploy your website
5. Your website will be available at https://liangzhang-keepmoving.github.io

## Step 4: Verify Deployment
1. Open your browser and navigate to https://liangzhang-keepmoving.github.io
2. Verify that your website is displayed correctly
3. Check the "GitHub Pages" section in your repository settings for any build errors

## Updating Your Website
To update your website in the future:
1. Make changes to your local files
2. Commit the changes:
   ```bash
   git add .
   git commit -m "Update website"
   ```
3. Push the changes to GitHub:
   ```bash
   git push origin main
   ```
4. GitHub Pages will automatically rebuild and deploy your changes within a few minutes

## Troubleshooting
- If you see a 404 error, make sure:
  - Your repository name is exactly `liangzhang-keepmoving.github.io`
  - You have pushed your code to the main branch
  - You have enabled GitHub Pages in the repository settings
  - You have waited a few minutes for the deployment to complete

- If you encounter Git authentication issues, consider setting up SSH keys:
  - https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account

## Technical Details
- Your website is built with HTML, CSS, and JavaScript
- It uses responsive design for mobile and desktop devices
- The website includes:
  - Home page with brief introduction
  - Blog page with articles
  - About Me page with detailed information about your background, skills, and achievements

## License
© 2024 Liang Zhang. All rights reserved.