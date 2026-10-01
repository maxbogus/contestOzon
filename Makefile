# contest - Backlog Makefile
SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
.PHONY := help list ready next rice status gate inventory inventory-json archive suggestions test lint

help: ## Show this help
	@python3 scripts/help.py

list: ## All tickets by status + RICE top-3 ready
	@python3 scripts/check_plan.py

ready: ## Top-3 ready tickets by RICE
	@python3 scripts/check_plan.py --next

next: ## Single next ticket
	@python3 scripts/check_plan.py --next -n 1

rice: ## All backlog sorted by RICE
	@python3 scripts/check_plan.py --rice

status: ## Single ticket info (make status ID=T1.1.1)
	@python3 scripts/check_plan.py --status $(ID)

gate: ## Check ticket gate (make gate ID=T1.1.1)
	@python3 scripts/check_plan.py --gate $(ID)

inventory: ## LLM-optimized backlog profile
	@python3 scripts/inventory.py

inventory-json: ## JSON inventory
	@python3 scripts/inventory.py --json

archive: ## Archive done ticket (make archive ID=T1.0.1)
	@python3 scripts/check_plan.py --archive $(ID)

suggestions: ## AI agent prompts
	@python3 scripts/check_plan.py --suggestions

test: ## Run tool tests
	@python3 -m pytest scripts/tests/ -v

lint: ## Lint scripts
	@python3 -m ruff check scripts/ 2>/dev/null || echo 'ruff not installed'

