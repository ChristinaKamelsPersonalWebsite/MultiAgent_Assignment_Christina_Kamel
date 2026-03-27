to run method 1 : we need 4 commands

docker network create --driver bridge agent_net
docker run --name redis --network agent_net -p 6379:6379 -v redis_data:/data -d redis:7-alpine
docker build -t pv_agent:latest .
docker run --name pv_agent --network agent_net --env-file .env -v %cd%\data:/app/data -e REDIS_HOST=redis -it pv_agent:latest



to run method 2 : we need 1 command (replaces all the manual commands with (dockerfile+dockercompose) + 1 command)

docker compose up --build