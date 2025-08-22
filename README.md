# Liang Zhang's Personal Website

This is a personal website built with Hugo and the PaperMod theme.

## Project Structure

```
├── .github/           # GitHub workflows for deployment
├── .gitignore         # Git ignore rules
├── .gitmodules        # Git submodules (for theme)
├── content/           # Website content in Markdown
│   ├── _index.md      # Homepage content
│   ├── about.md       # About page
│   ├── posts/         # Blog posts
│   ├── project.md     # Projects page
│   └── publications.md # Publications page
├── fix_all_paths.py   # Tool to fix path references in HTML files
├── hugo.toml          # Hugo configuration file
├── static/            # Static assets
│   ├── CV_zhangliang25jlv2_0.pdf # CV
│   └── images/        # Images directory
├── themes/            # Hugo themes
│   └── PaperMod/      # PaperMod theme (git submodule)
└── verify_menu_content.py # Tool to verify menu items match content structure
```

## Getting Started

### Prerequisites
- Install Hugo: https://gohugo.io/getting-started/installing/

### Running the site locally
```bash
hugo server -D
```

### Building the site
```bash
hugo
```

The built site will be generated in the `public/` directory.

## Tools

### fix_all_paths.py
This script fixes path references in all HTML files in the `public` directory.
It ensures:
- Navigation menus contain the correct menu items
- References to '/research/' are replaced with '/project/'
- Canonical links and metadata are properly set

Run with:
```bash
python fix_all_paths.py
```

### verify_menu_content.py
This script verifies that each menu item in `hugo.toml` has a corresponding content file in the `content` directory.

Run with:
```bash
python verify_menu_content.py
```

## Deployment
This site is automatically deployed to GitHub Pages via the GitHub workflow in `.github/workflows/gh-pages.yml`.