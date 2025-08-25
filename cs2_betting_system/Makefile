.PHONY: up down restart logs build clean dev prod backup restore
up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

build:
	docker compose build

clean:
	docker compose down -v
	docker system prune -af

dev:
	docker compose --profile dev up -d

prod:
	docker compose -f docker-compose.yml up -d

backup:
	docker exec cs2_postgres pg_dump -U cs2admin cs2_betting > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore:
	docker exec -i cs2_postgres psql -U cs2admin cs2_betting < $(FILE)
