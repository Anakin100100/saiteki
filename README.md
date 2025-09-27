# Description

ShinkaEvolve has achieved a SOTA result on circle packing, an optimization problem that has been challenging computer scientists for decated, for 12 dollars in LLM API costs. This is a paradigm defining moment for optimizing software. This project makes shinka available to teams as a web app. 

# Contribution guide
Contributions to the project are welcome from everybody. You can look for available tasks in the associated project https://github.com/users/Anakin100100/projects/2 but don't worry if there aren't any tasks that you feel comfortable starting on. There's a lot to work on so not every potential improvemnt has a task. Before contributing please schedule a short meeting with me at https://cal.com/paweł-biegun-lbvvws to discuss the project, your vision for it and how you would like to contribute. 

# Setup
This project utilizes several dependencies running inside of Docker so you are going to need to install it togerther with docker compose https://docs.docker.com/compose/install/ 

For managing Python packages we're using uv. It is recommended to install it through the standalone installer https://docs.astral.sh/uv/getting-started/installation/#standalone-installer 

For javascript package management for the backend and frontend we're using bun https://bun.com/docs/installation 

# Running the project
Start the containers
```bash
docker compose up -d 
```
Install packages and run migrations
```bash
cd saiteki-web
bun install
bun db:push
```

Run dev server for backend and frontend
```bash
bun dev
```

Install python packages
```bash
cd saiteki-worker
uv sync
```

Expose your OpenAI key in the worker shell 
```bash
export OPENAI_API_KEY='<your key>'
```

Run the worker
```bash
uv run main.py
```

# Trademark and affiliation notice

This project is not affiliated with Sakana AI. ShinkaEvolve and Sakana AI are trademarks belonging Sakana AI Co, Ltd. All trademarks belong to their respective owners.
