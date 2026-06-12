#!/usr/bin/env bash

#ssh -R 17897:127.0.0.1:7897 Ascend
#!export http_proxy=http://127.0.0.1:17897
#export https_proxy=http://127.0.0.1:17897
#export HTTP_PROXY=http://127.0.0.1:17897
#export HTTPS_PROXY=http://127.0.0.1:17897
#set -euo pipefail

export ANTHROPIC_AUTH_TOKEN="sk-82b25066ee574b5993ace1dc5490fd2c"
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export ANTHROPIC_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_EFFORT_LEVEL="max"
export CLAUDE_CONFIG_DIR="$HOME/.claude"

