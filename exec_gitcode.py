#!/usr/bin/env python3
"""
exec_gitcode.py
===============
Execute Python code directly from GitHub URL (regular or raw).
Converts github.com URLs to raw.githubusercontent.com automatically.

Usage: 
    python exec_gitcode.py <github_url> [script_args...]
    
Examples:
    # Regular GitHub URL (converted to raw automatically)
    python exec_gitcode.py https://github.com/sudhir-voleti/rfmpaper/blob/main/quick_check_idata.py results/*.pkl
    
    # Raw URL (works as-is)
    python exec_gitcode.py https://raw.githubusercontent.com/sudhir-voleti/rfmpaper/main/quick_check_idata.py results/*.pkl
"""
import sys
import urllib.request
import tempfile
import os
import re


def to_raw_url(url):
    """
    Convert regular GitHub URL to raw.githubusercontent URL.
    
    Handles:
    - https://github.com/user/repo/blob/branch/file.py
    - https://github.com/user/repo/blob/main/file.py
    - https://raw.githubusercontent.com/user/repo/branch/file.py (pass-through)
    """
    # Already raw?
    if 'raw.githubusercontent.com' in url:
        return url
    
    # Convert github.com/blob/ to raw
    # Pattern: github.com/user/repo/blob/branch/file.py
    pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)'
    match = re.match(pattern, url)
    
    if match:
        user, repo, branch, filepath = match.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"        
        print(f"Converted to raw: {raw_url}")
        return raw_url
    
    # Try alternative pattern with /tree/ (also common)
    pattern_tree = r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)'
    match_tree = re.match(pattern_tree, url)
    
    if match_tree:
        user, repo, branch, filepath = match_tree.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"
        print(f"Converted to raw: {raw_url}")
        return raw_url
    
    # Unrecognized format, try as-is
    print(f"Warning: Unrecognized GitHub URL format, trying as-is")
    return url


def exec_from_github(url, args):
    """Fetch Python code from GitHub and execute."""
    raw_url = to_raw_url(url)
    
    print(f"Fetching: {raw_url}")
    
    # Download
    req = urllib.request.Request(raw_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            code = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print(f"Error fetching: {e}")
        print("Check that the URL is correct and the file exists.")
        sys.exit(1)
    
    # Check if we got HTML instead of code (wrong URL)
    if code.strip().startswith('<!DOCTYPE') or code.strip().startswith('<html'):
        print("Error: Received HTML instead of Python code.")
        print("Please check the GitHub URL is correct.")
        sys.exit(1)
    
    print(f"Downloaded {len(code)} bytes")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Execute with args
        sys.argv = [temp_path] + args
        with open(temp_path) as f:
            exec(compile(f.read(), temp_path, 'exec'), 
                 {'__name__': '__main__', '__file__': temp_path})    
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Execute Python code directly from GitHub")
        print()
        print("Usage: python exec_gitcode.py <github_url> [script_args...]")
        print()
        print("Examples:")
        print("  # Regular GitHub URL (auto-converted to raw):")
        print("  python exec_gitcode.py https://github.com/user/repo/blob/main/script.py arg1")
        print()
        print("  # Your specific use case:")
        print("  python exec_gitcode.py https://github.com/sudhir-voleti/rfmpaper/blob/main/quick_check_idata.py results/*.pkl")
        sys.exit(1)
    
    url = sys.argv[1]
    script_args = sys.argv[2:]
    
    exec_from_github(url, script_args)
