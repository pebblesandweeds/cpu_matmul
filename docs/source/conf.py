# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pebbles and Weeds Blog'
copyright = '2024, Pebbles and Weeds'
author = 'Pebbles and Weeds'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Adding `sphinxcontrib.plantuml` to list of extensions
extensions = [
    'sphinxcontrib.plantuml',
    "sphinx.ext.githubpages",
]

# PlantUML jar file
plantuml = 'java -jar /opt/homebrew/Cellar/plantuml/1.2024.6/libexec/plantuml.jar'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'

html_theme_options = {
    'logo': 'pebblesandweeds.png',  # Reference your logo
    'github_user': 'pebblesandweeds',
    'github_repo': 'cpu_matmul',
    'description': 'A detailed guide to CPU Matrix Multiplication in C',
    'show_powered_by': False,
    'show_related': False,
    'note_bg': '#FFF59C',
}

# Sidebar only on index page, not on blog pages
html_sidebars = {
    'index': ['about.html', 'navigation.html'],  # Show sidebar components only on index.html
    '**': []  # No sidebar on all other pages
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',  # Ensure this matches the path where your CSS is located
]

# SEO Metadata
html_meta = {
    "description": "Pebbles and Weeds Blog: Exploring data science, AI, programming, and high-performance computing",
    "keywords": "data science, AI, machine learning, GPU computing, C programming, high-performance computing",
    "author": "Pebbles and Weeds",
    "robots": "index, follow"
}

# Sitemap
extensions.append('sphinx_sitemap')
html_baseurl = 'https://blog.pebblesandweeds.com/'  # Replace with your actual URL
sitemap_filename = 'sitemap.xml'
