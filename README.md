# FundGraph on AWS

Ask a mutual fund database in plain English. Graph RAG over Amazon RDS for PostgreSQL,
with the schema graph in Amazon Neptune and the models on Amazon Bedrock. No Docker.

## Stack
| Piece | Service |
|---|---|
| Fund data (`mf_data`) and agent data (`agent_meta`) | Amazon RDS for PostgreSQL, with pgvector |
| Schema graph | Amazon Neptune (openCypher) |
| Chat model, embeddings, prompt-attack check | Amazon Bedrock |
| Monitoring | CloudWatch and AWS X-Ray |
| App | FastAPI + LangGraph, run locally |

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then fill it in
    pytest tests/ -v
