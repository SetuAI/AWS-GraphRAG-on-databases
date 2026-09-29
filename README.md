# FundGraph on AWS: ask a mutual fund database in plain English

**`aws-fundgraph-rag-on-db`** · Graph RAG over a relational database · Amazon RDS for PostgreSQL · pgvector · Amazon Neptune · Amazon Bedrock · Bedrock Guardrails · LangGraph · FastAPI · pip

Ask a question about a mutual fund company's data in plain English. The system works out which tables hold the answer and how to join them, including links that were never declared in the database. It writes safe, read-only SQL, runs it, and returns:

- the answer as a **table**
- a **plain-English explanation** of what the numbers show
- the **exact SQL** it ran, and **every safety check** the SQL passed

```
"Why did net flows in Sahyadri Credit Risk Fund drop in March 2024?"
        │
        ▼   ~15–40 seconds
┌──────────────────────────────────────────────────────────────────────────────┐
│ month       net_flow        Net flows were positive through February 2024,   │
│ 2024-01     +5,130,000      then turned sharply negative in March, in the     │
│ 2024-02     +5,140,000      same month as the credit rating of a bond the     │
│ 2024-03    -21,698,609      fund holds was cut from AA- to BBB-.              │
│ 2024-04     -4,170,000                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
  + the SQL, the joins it used, and every safety check that passed
```

**No Docker.** The data lives in Amazon RDS, the graph in Amazon Neptune, the models on Amazon Bedrock. The application runs on your own machine.

---

## Contents

1. [The problem](#1-the-problem)
2. [What this project builds](#2-what-this-project-builds)
3. [Architecture](#3-architecture)
4. [Technology](#4-technology)
5. [Project structure](#5-project-structure)
6. [The two databases](#6-the-two-databases)
7. [The data: 20 tables](#7-the-data-20-tables)
8. [How the system learns the database](#8-how-the-system-learns-the-database)
9. [How a question is answered: 11 steps](#9-how-a-question-is-answered-11-steps)
10. [Safety: five layers](#10-safety-five-layers)
11. [Evaluation](#11-evaluation)
12. [Setup: the AWS side](#12-setup-the-aws-side)
13. [Setup: your machine](#13-setup-your-machine)
14. [Running the pipeline](#14-running-the-pipeline)
15. [Using it](#15-using-it)
16. [Tests and gates](#16-tests-and-gates)
17. [Configuration reference](#17-configuration-reference)
18. [Cost, and keeping it low](#18-cost-and-keeping-it-low)
19. [Troubleshooting](#19-troubleshooting)
20. [Using it on your own database](#20-using-it-on-your-own-database)

---

## 1. The problem

A mutual fund company keeps its data in a relational database: schemes, prices, holdings, investors and every transaction. When someone in the business has a question, today it goes like this:

```
Business user asks a question
  → sends it to the data team
  → someone works out which tables hold the answer
  → someone writes SQL that joins them correctly
  → someone runs it, checks it, puts it in a spreadsheet
  → the answer comes back, often days later
```

Two things make this slow:

1. **Few people can write the SQL.** A real fund database has 50+ tables. Knowing which ones to join, and on which columns, takes someone who knows the database well.
2. **Some links between tables are not written down.** They exist in the data, but no foreign key declares them. Only experienced people know them.

## 2. What this project builds

A system that turns a plain-English question into a checked answer in under a minute, and **shows its work** at every step. Two goals, in this order:

1. **Prove it works**, on a realistic synthetic copy of a fund database.
2. **Be replicable**: every design choice favours what someone else can rebuild on their own database, and every file is commented for someone new to it.

## 3. Architecture

```
  YOUR MACHINE                                  AWS  (the region in AWS_BEDROCK_ARN)
  ─────────────────────────────────             ──────────────────────────────────────
  Browser: demo page, graph page
     │
  FastAPI  (uvicorn api.main:app)
     │
  LangGraph agent, 11 steps
     │  1  guard the question  ───────────────▶  Bedrock Guardrail
     │  2  understand it       ───────────────▶  Bedrock: chat model
     │  3  match names         ──┐
     │  4  pick tables         ──┴───────────▶  RDS: agent_meta (pgvector) + Titan
     │  5  find the joins      ───────────────▶  Neptune: 20 nodes, 22 join edges
     │  6  write the SQL       ───────────────▶  Bedrock: chat model
     │  7  check the SQL       ──┐
     │  8  run it, read only   ──┴───────────▶  RDS: mf_data (read only)
     │  9  summarise
     │ 10  explain             ───────────────▶  Bedrock: chat model
     │ 11  remember
     │
  evals/run_eval.py  → scores 18 questions
```

**Everything that works out *how* to answer uses `agent_meta` and Neptune. Only step 8 touches the fund data, and only to read.**

## 4. Technology

| Job | Tool | Where |
|---|---|---|
| Understanding questions, table notes, SQL, commentary | The Bedrock model in **`AWS_MODEL_BEDROCK_ID`** (Nova Pro by default), via the **Converse API** | `metadata/llm.py` |
| Embeddings (text → 1,024 numbers) | **Titan Text Embeddings V2** on Bedrock | `metadata/llm.py` |
| Prompt attack detection | **Amazon Bedrock Guardrails**, prompt-attack filter | `guardrails/input_guard.py` |
| Fund data and agent data | **Amazon RDS for PostgreSQL 17**, two databases on one instance | `adapters/postgres.py` |
| Vector search | **pgvector**, HNSW index, cosine distance | `metadata/store.py` |
| Graph of tables and joins | **Amazon Neptune**, openCypher over signed HTTPS | `adapters/graph_neptune.py` |
| Agent workflow | **LangGraph**, one graph, fixed steps | `agent/graph.py` |
| SQL parsing and checks | **sqlglot** | `guardrails/sql_checks.py` |
| Web API and pages | **FastAPI** + Uvicorn | `api/main.py` |
| Access | **AWS IAM** user with a least-privilege policy | `infra/aws/iam-policy.json` |
| Cost alerts | **AWS Budgets** | `infra/aws/budget*.json` |
| Synthetic data | **numpy** + **polars** | `dataplane/generator/` |
| Packages | **pip** and a virtual environment (Python 3.12+) | `requirements.txt` |
| Tests, lint, evaluation | **pytest**, **ruff**, and `evals/` | `tests/`, `evals/` |

**The Converse API matters more than it looks.** Every chat model on Bedrock takes the same request format, which is why switching from Claude to Nova during this build was two lines in `.env` and no code change at all.

## 5. Project structure

```
aws-fundgraph-rag-on-db/
├── .env                       your keys and addresses. NEVER committed
├── .env.example               the same names, blank. Copy to .env
├── .gitignore                 keeps .env, .venv/ and caches out of Git
├── requirements.txt           every package (pip install -r requirements.txt)
├── pyproject.toml             pytest and ruff settings only
│
├── infra/aws/                 files the one-time AWS commands read
│   ├── iam-policy.json            Bedrock, guardrail, Neptune permissions
│   ├── guardrail-content-policy.json  the PROMPT_ATTACK filter
│   └── budget.json, budget-notifications.json
│
├── config/                    every rule and limit (never any answers)
│   ├── models.yaml                embedding model and size, reply length
│   ├── agent.yaml                 retries, timeout, row cap, cost limit
│   ├── metrics.yaml               8 approved business definitions
│   ├── metadata.yaml              link, embedding and graph rules
│   ├── pii.yaml                   personal-data columns
│   └── generator.yaml             seed, scale, date window
│
├── scripts/
│   ├── rds_setup.py               creates agent_meta, switches on pgvector
│   └── allow_my_ip.sh             re-opens both firewalls when your IP changes
│
├── dataplane/                 creates the synthetic data (delete for a real one)
│   ├── schema/                    4 SQL files + apply_schema.py
│   ├── generator/                 run.py + 7 generator files
│   └── load/copy_loader.py        fast COPY into RDS
│
├── adapters/
│   ├── postgres.py                the ONLY code that talks to the fund database
│   └── graph_neptune.py           the ONLY code that talks to Neptune
│
├── metadata/                  teaches the system the database
│   ├── build.py / review.py / publish.py    the three steps
│   ├── links.py / notes.py                  find links, draft notes
│   ├── retrieval.py                         search tables, match names, find joins
│   ├── store.py                             the meta_ tables (vector(1024))
│   └── llm.py                               the ONLY code that calls Bedrock
│
├── agent/                     the LangGraph workflow
│   ├── graph.py, context.py
│   └── steps/understand.py, build_query.py, answer.py
├── guardrails/                input_guard.py, sql_checks.py
├── api/                       main.py, telemetry.py, static/index.html, static/graph.html
│
├── evals/                     how good are the answers?
│   ├── run_eval.py                the one file you run
│   ├── questions/questions.yaml   18 questions with their ground truth
│   ├── metrics/retrieval.py       precision, recall, MRR, join path
│   ├── metrics/generation.py      groundedness, completeness, refusals
│   ├── metrics/judge.py           optional: a model's opinion
│   ├── metrics/ragas_adapter.py   optional: RAGAS
│   └── answer_key/                the correct answers (never read by the pipeline)
│
└── tests/                     one file per stage, 80 tests
```

**Four rules the structure follows:**

1. **`config/` holds rules; `evals/` holds answers.** The pipeline reads `config/` and never reads `evals/`, so its scores mean something.
2. **One file per outside system.** `adapters/postgres.py` for RDS, `adapters/graph_neptune.py` for Neptune, `metadata/llm.py` for Bedrock. Changing any one of them means changing one file.
3. **`dataplane/` is disposable.** It only creates test data. On a real database it is deleted, and nothing else changes.
4. **Nothing unreviewed is used.** Notes start as `draft`, discovered links as `proposed`, and only `approved` items reach the model or the graph.

## 6. The two databases

Both are databases on the **same RDS instance**, with separate connection settings in `.env`.

| | `mf_data` | `agent_meta` |
|---|---|---|
| **Holds** | The fund data (stands in for a client's database) | What the agent knows *about* the fund data |
| **Contents** | 20 tables, 1,132,667 rows, ~210 MB | `meta_tables` (20), `meta_columns` (124), `meta_edges` (22), `meta_embeddings` (1,415) |
| **Agent access** | **Read only** | Read and write |
| **Extensions** | None needed | pgvector |
| **Rebuilt when** | Never, by the agent | Any time: build → review → publish |

**Why two:** a client's production database must not get new tables or extensions, and may not even be PostgreSQL. The agent needs write access for its notes and links, but must never write to fund data. A separate `agent_meta` solves both, and it is only a few MB.

**Why the graph is not in either:** on the Azure version the graph lived in Postgres, using the Apache AGE extension. **RDS does not offer AGE**, so on AWS the graph moved to Neptune. That is the one architectural difference, and it affects three files.

## 7. The data: 20 tables

A fictitious fund house, **Sahyadri Mutual Fund**, with 40 schemes, over 5 years: **1 Sep 2021 to 31 Aug 2026**. All company, index and agency names are invented. Every number is created by code from a fixed seed; no model creates any data.

| Group | Tables | Rows |
|---|---|---:|
| **Fund house** | `scheme_categories`, `benchmarks`, `benchmark_values`, `schemes`, `scheme_plans`, `fund_managers`, `scheme_manager_assignments`, `nav_history`, `scheme_aum_monthly` | ~212,000 |
| **Market** | `sectors`, `issuer_groups`, `issuers`, `securities`, `credit_rating_history`, `portfolio_holdings` | ~115,000 |
| **Investors** | `distributors`, `investors`, `folios`, `sip_registrations` | ~122,000 |
| **Transactions** | `transactions` (split into 60 monthly partitions) | ~683,000 |

**Built in on purpose, to match a real client database:**

| Feature | Detail |
|---|---|
| **3 hidden links** (no foreign key) | `scheme_aum_monthly.scheme_id → schemes.scheme_id` (same name) · `credit_rating_history.sec_cd → securities.security_code` (different names) · `transactions.arn_code → distributors.arn_no` (different names) |
| **2 old-style tables** | `credit_rating_history` (`rtg_dt`, `outlk`, `sec_cd`) and `distributors` (`arn_no`, `dist_nm`, `actv_flg`) |
| **Personal data, masked at creation** | PAN `ABCPX****F`, masked email, mobile and bank account. Listed in `config/pii.yaml` |
| **Two routes to AUM** | `scheme_aum_monthly.aum_amount` and the sum of `portfolio_holdings.market_value`. They agree exactly; `metrics.yaml` names the approved one |
| **Consistency** | Units × NAV = amount; no account below zero units; holdings add up to 100% |

**6 planted patterns**, so demo questions have real answers:

| # | Pattern | Question it answers |
|---|---|---|
| 1 | *Vardhaman Housing Finance* bond cut AA- → BBB- on 12 Mar 2024; redemptions spike in the two schemes holding it | Why did Credit Risk Fund net flows drop in March 2024? |
| 2 | Mid Cap Fund holds ~24% in five **Trident Group** companies | Which scheme has the most exposure to one group? |
| 3 | Focused Fund changes manager on 3 Oct 2023, then starts beating its benchmark | Did performance change after the manager changed? |
| 4 | Tier-3 city SIP cancellations rise 3.5× in the last 12 months | Where are SIP cancellations rising? |
| 5 | **Konkan Wealth Partners** has ~17% rejected transactions vs ~2% | Which distributor has the highest rejection rate? |
| 6 | B&FS Fund holds 12.4% in **Malabar Commercial Bank** from Mar 2026 | Is any scheme above the 10% single-issuer limit? |

The answers are recorded in `evals/answer_key/`, which the pipeline never reads.

## 8. How the system learns the database

Three commands, writing only to `agent_meta` and Neptune.

```
mf_data (read only)
   │
   ├─ read catalog     tables, columns, types, keys            adapters/postgres.py
   ├─ read statistics  distinct counts, null rates, samples    (never PII values)
   │
   ├─ find links       1. declared foreign keys               → approved
   │                   2. same column name + value check      → proposed
   │                   3. different names, values overlap     → proposed    metadata/links.py
   │
   ├─ draft notes      the chat model describes every table and column → draft
   │                                                                       metadata/notes.py
   │   ── metadata.build ──────────────────────────────────────────────────
   │
   ├─ review           a person reads every note and link, rejects any
   │                   that are wrong, approves the rest                    metadata/review.py
   │   ── metadata.review ─────────────────────────────────────────────────
   │
   └─ publish          approved notes + values → Titan → pgvector
                       approved links          → the Neptune graph          metadata/publish.py
       ── metadata.publish ────────────────────────────────────────────────
```

**Nothing unreviewed reaches the SQL model.**

**What gets embedded:** each table's note, plus the values of text columns with fewer than 10,000 distinct values: scheme, issuer, group, sector, manager and distributor names. PII columns and code-like columns (`*_code`, `*_cd`, `isin`, `*_no`) are never embedded.

**Only changed tables are redrafted.** Each table gets a fingerprint of its columns and types, so a second run costs almost nothing on a large database. Force a redraft with `--redraft`.

**Name matching is hybrid, not pure vector search.** A question typing "vardhaman housing" must reach *Vardhaman Housing Finance Ltd*, not *Vardhaman Holdings Ltd*. Embeddings compare overall meaning, and on short company names two similar names sit almost on top of each other, so `retrieval.py` fetches a shortlist by meaning and re-ranks it counting the words actually typed. `VECTOR_WEIGHT = 0.6` keeps meaning in charge while letting exact words break a near-tie.

## 9. How a question is answered: 11 steps

One LangGraph workflow, in a fixed order, so every question follows the same auditable path.

| # | Step | What it does | Uses |
|---|---|---|---|
| 1 | `guard_input` | Checks the question for prompt attacks | Bedrock Guardrail |
| 2 | `understand` | Works out the intent, names, dates, approved metrics, and whether it's a request to change data | Chat model |
| 3 | `match_entities` | Matches loose names to exact values: "sahyadri credit fund" → *Sahyadri Credit Risk Fund* | Titan + pgvector + word overlap |
| 4 | `select_tables` | Picks tables: the ones the names and metrics need, plus vector search on table notes | Titan + pgvector |
| 5 | `find_join_path` | Finds the shortest route connecting those tables | **Neptune** |
| 6 | `write_sql` | Writes one SELECT from the approved notes, joins, exact names and metric definitions | Chat model |
| 7 | `check_sql` | SELECT only, known tables, a real date range, estimated cost under the limit | sqlglot, EXPLAIN |
| 8 | `run_sql` | Runs it read-only, 15 s timeout, 5,000-row cap | `mf_data` |
| 9 | `summarise` | Row count, totals, first 20 rows. This is all the model sees of the result | Python |
| 10 | `write_commentary` | Explains the result using only numbers from the summary; says "in the same month as", never "because of" | Chat model |
| 11 | `remember` | Saves the turn so follow-ups work | LangGraph memory |

```
guard_input ──attack──────────────────────────────────────────────→ blocked
understand  ──change data / off topic─────────────────────────────→ blocked
            ──impossible to answer────────────────────────────────→ needs_clarification
write_sql   ──model refuses───────────────────────────────────────→ blocked
check_sql / run_sql ──fails──→ back to write_sql (max 2 retries) ──→ failed
everything passes ────────────────────────────────────────────────→ completed
```

**How the join route is worked out:** the 22 edges are read from Neptune **once**, when the first question arrives, then searched in Python with a breadth-first search: fewest joins first, most confident links breaking ties. One request per server start beats one Cypher path query per pair of tables, and the result is identical.

**How JSON comes back:** Converse has no "JSON only" switch. Every prompt asks for a JSON object, and `metadata/llm.py` parses from the first `{` to the last `}`, so a reply wrapped in a sentence still works. `temperature` is 0.

**Units are never converted by the model.** The commentary prompt requires rupee figures to be written exactly as they appear in the result. Models are unreliable at arithmetic: during this build one consistently divided by a million and called the result crore, making every figure ten times too big. Removing the conversion removes the whole error class.

## 10. Safety: five layers

| Layer | What it stops | Where |
|---|---|---|
| 1. **Bedrock Guardrail** | Attempts to override the AI's instructions, before any model sees them | `guardrails/input_guard.py` |
| 2. **Request type** | Requests to delete, update or create data, refused before any SQL is written | `agent/steps/understand.py` |
| 3. **SQL checks** | Anything but one SELECT; unknown tables; `transactions` queries without a real date range | `guardrails/sql_checks.py` |
| 4. **Cost check** | Queries PostgreSQL estimates as too expensive, via EXPLAIN, without running them | `agent/steps/build_query.py` |
| 5. **Read-only session** | PostgreSQL itself refuses any write; plus a 15 s timeout and a 5,000-row cap | `adapters/postgres.py` |

**Least-privilege access:** the IAM user can call Bedrock models, apply the guardrail, and query Neptune. It cannot create, delete or read any other AWS resource, and cannot change security groups.

**Personal data** is never sent to Bedrock, never embedded, and is masked in the data itself.

**Causation:** the commentary describes timing, never cause, because the data shows only when things happened.

## 11. Evaluation

Tests answer *"did I break something?"*. The evaluation answers *"is it any good?"*.

```bash
python -m evals.run_eval
```

18 questions with known answers: the 6 planted stories, 6 straightforward ones, 3 that need hidden links, and 3 that **must be refused**. Each is scored on finding, answering and refusing.

```
18 questions · apac.amazon.nova-pro-v1:0 · 1m 7s
==========================================================================
FINDING     precision@5 0.62   recall@5 0.91   MRR 0.95   join path 13/15
ANSWERING   grounded 14/15     complete 14/15   SQL ran 14/15   (SQL retries: 1)
REFUSING    3/3
OVERALL     status correct 18/18
```

| Number | Meaning |
|---|---|
| `recall@5` | Of the tables needed, how many were found. **The one to watch:** a missing table means the question cannot be answered |
| `precision@5` | Of the tables chosen, how many were needed. Spare tables are untidy, not fatal |
| `join path` | How many questions used every required join, counted from the graph route **or** the SQL that ran |
| `grounded` | How many explanations quoted only figures traceable to the rows, their column totals, or the row count |
| `complete` | How many mentioned the fund, period or metric the question asked about |
| `REFUSING` | Whether the three questions that must be refused were refused |

**Options:** `--only p1` for one question, `--judge` to add a model's opinion of the wording, `--ragas` for RAGAS.

**The groundedness check is arithmetic, not opinion.** It pulls every number out of the explanation and looks for it in the rows, allowing rounding, lakh and crore, column totals and the row count, and ignoring years and numbers that came from the question or the SQL. That is what catches an invented figure, and it gives the same answer every run.

**Results are saved** to `evals/results/<timestamp>.json`, so two runs can be compared. With 18 questions, one question flipping moves a score by about 0.06: treat smaller changes as noise.

## 12. Setup: the AWS side

Run these once, in **AWS CloudShell** (the `>_` icon in the console), as an account administrator. Everything lives in one region; these examples use `ap-south-1`.

```bash
export AWS_REGION=ap-south-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

**1. A budget, before anything that can bill:**

```bash
aws budgets create-budget --account-id "$ACCOUNT_ID" \
  --budget file://infra/aws/budget.json \
  --notifications-with-subscribers file://infra/aws/budget-notifications.json
```

**2. The network and a security group for the database:**

```bash
VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query "Subnets[].SubnetId" --output text)

SG_ID=$(aws ec2 create-security-group --group-name fundgraph-db-sg \
  --description "FundGraph RDS: PostgreSQL from my laptop only" \
  --vpc-id "$VPC_ID" --query GroupId --output text)

# YOUR machine's address, from: curl -s https://checkip.amazonaws.com
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 5432 --cidr "YOUR.IP/32"
```

`/32` means that one address. **Never `0.0.0.0/0`.**

**3. The database:**

```bash
read -s -p "Database password (letters and numbers only): " DB_PASS; echo
PG_VERSION=$(aws rds describe-db-engine-versions --engine postgres \
  --query "DBEngineVersions[?starts_with(EngineVersion,'17.')].EngineVersion | [-1]" --output text)

aws rds create-db-instance --db-instance-identifier fundgraph-db \
  --engine postgres --engine-version "$PG_VERSION" \
  --db-instance-class db.t4g.micro --allocated-storage 20 --storage-type gp2 --no-multi-az \
  --master-username fundgraph --master-user-password "$DB_PASS" \
  --db-name mf_data --publicly-accessible --vpc-security-group-ids "$SG_ID" \
  --backup-retention-period 1 --no-enable-performance-insights --no-deletion-protection

aws rds wait db-instance-available --db-instance-identifier fundgraph-db
aws rds describe-db-instances --db-instance-identifier fundgraph-db \
  --query "DBInstances[0].Endpoint.Address" --output text
```

**`--engine postgres`, not Aurora.** They sit next to each other in the console and only one is free-tier eligible. **`--db-name mf_data`** is easy to miss; without it you get a server with no database in it.

**4. Bedrock: the model and the guardrail:**

```bash
aws bedrock list-inference-profiles \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId,'nova')].[inferenceProfileId,inferenceProfileArn]" \
  --output table

cat > /tmp/prompt-attack.json <<'EOF'
{"filtersConfig":[{"type":"PROMPT_ATTACK","inputStrength":"HIGH","outputStrength":"NONE"}]}
EOF

GUARDRAIL_ID=$(aws bedrock create-guardrail --name fundgraph-input-guard \
  --description "Blocks prompt attacks on the FundGraph agent" \
  --content-policy-config file:///tmp/prompt-attack.json \
  --blocked-input-messaging "This request was blocked by the input guardrail." \
  --blocked-outputs-messaging "This response was blocked by the guardrail." \
  --query guardrailId --output text)

aws bedrock create-guardrail-version --guardrail-identifier "$GUARDRAIL_ID" --query version --output text
```

The model's **ID** and **ARN** go into `.env`; the code reads the region from the ARN.

**Anthropic models need a one-time use-case form** in the Bedrock console, and are sold through AWS Marketplace, so the account needs a payment method Marketplace accepts. **Amazon's own models (Nova, Titan) avoid Marketplace entirely**, which is why this build uses Nova Pro.

**5. The application's IAM user:**

```bash
aws iam create-user --user-name fundgraph-app
aws iam put-user-policy --user-name fundgraph-app --policy-name fundgraph-bedrock \
  --policy-document file:///tmp/fundgraph-policy.json
aws iam create-access-key --user-name fundgraph-app \
  --query "AccessKey.[AccessKeyId,SecretAccessKey]" --output text
```

The secret is shown once. It goes into `.env` and nowhere else.

**6. Neptune:**

```bash
aws neptune create-db-subnet-group --db-subnet-group-name fundgraph-neptune-subnets \
  --db-subnet-group-description "FundGraph graph" --subnet-ids $SUBNETS

NEPTUNE_SG=$(aws ec2 create-security-group --group-name fundgraph-neptune-sg \
  --description "FundGraph Neptune: port 8182 from my laptop" \
  --vpc-id "$VPC_ID" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id "$NEPTUNE_SG" \
  --protocol tcp --port 8182 --cidr "YOUR.IP/32"

aws neptune create-db-cluster --db-cluster-identifier fundgraph-graph --engine neptune \
  --db-subnet-group-name fundgraph-neptune-subnets --vpc-security-group-ids "$NEPTUNE_SG" \
  --enable-iam-database-authentication --storage-type standard --backup-retention-period 1

aws neptune create-db-instance --db-instance-identifier fundgraph-graph-1 \
  --db-cluster-identifier fundgraph-graph --engine neptune \
  --db-instance-class db.t3.medium --publicly-accessible

aws neptune wait db-instance-available --db-instance-identifier fundgraph-graph-1
aws neptune describe-db-clusters --db-cluster-identifier fundgraph-graph \
  --query "DBClusters[0].[Endpoint,DbClusterResourceId]" --output table
```

Then allow the application to query it, using that **resource ID** (not the cluster name):

```bash
cat > /tmp/neptune-policy.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Action":["neptune-db:connect","neptune-db:ReadDataViaQuery",
           "neptune-db:WriteDataViaQuery","neptune-db:DeleteDataViaQuery"],
 "Resource":"arn:aws:neptune-db:ap-south-1:${ACCOUNT_ID}:${RESOURCE_ID}/*"}]}
EOF
aws iam put-user-policy --user-name fundgraph-app --policy-name fundgraph-neptune \
  --policy-document file:///tmp/neptune-policy.json
```

**Neptune needs two separate things, and both must be right:** a security group rule for the network, and `neptune-db` permissions for the request. With only the first, the connection opens and the query is refused.

**`--enable-iam-database-authentication` is required** before Neptune allows public access at all.

## 13. Setup: your machine

```bash
git clone <your-repository-url> aws-fundgraph-rag-on-db
cd aws-fundgraph-rag-on-db

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
git check-ignore .env              # must print: .env
```

Then fill in `.env`:

```bash
AWS_ACCESS_KEY=AKIA...
AWS_SECRET_KEY=...
AWS_MODEL_BEDROCK_ID=apac.amazon.nova-pro-v1:0
AWS_BEDROCK_ARN=arn:aws:bedrock:ap-south-1:123456789012:inference-profile/apac.amazon.nova-pro-v1:0
BEDROCK_GUARDRAIL_ID=...
BEDROCK_VERSION=1
AWS_REGION=ap-south-1

DATA_DB_URL=postgresql://fundgraph:PASSWORD@fundgraph-db.xxxx.ap-south-1.rds.amazonaws.com:5432/mf_data
AGENT_DB_URL=postgresql://fundgraph:PASSWORD@fundgraph-db.xxxx.ap-south-1.rds.amazonaws.com:5432/agent_meta

NEPTUNE_CLUSTER_HOST=fundgraph-graph.cluster-xxxx.ap-south-1.neptune.amazonaws.com
NEPTUNE_PORT=8182
```

| Setting | Used for |
|---|---|
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` | Signing every AWS request, including Neptune |
| `AWS_MODEL_BEDROCK_ID` | The chat model, for every chat job |
| `AWS_BEDROCK_ARN` | **The region of every Bedrock call**, read from the ARN |
| `BEDROCK_GUARDRAIL_ID`, `BEDROCK_VERSION` | The prompt-attack check |
| `AWS_REGION` | Tracing. Must match the ARN's region; the code warns if not |
| `DATA_DB_URL`, `AGENT_DB_URL` | Same host and user, different database name |
| `NEPTUNE_CLUSTER_HOST`, `NEPTUNE_PORT` | The graph |

Then prove the AWS side works before anything else:

```bash
pytest tests/test_bedrock.py tests/test_guardrail.py -v     # 8 passed
```

## 14. Running the pipeline

```bash
# 1. Create the second database and switch on pgvector
python scripts/rds_setup.py
pytest tests/test_rds_setup.py -v            # 7 passed

# 2. Create the 20 empty tables
python dataplane/schema/apply_schema.py
pytest tests/test_stage1_schema.py -v        # 11 passed

# 3. Generate and load 1.1 million rows (1–5 minutes)
python -m dataplane.generator.run
pytest tests/test_stage2_data.py -v          # GATE 1: 13 passed

# 4. Check the graph database
pytest tests/test_neptune.py -v              # 4 passed

# 5. Read the database, find links, draft notes (1–3 min)
python -m metadata.build

# 6. Review: READ the notes and links, reject any that are wrong
python -m metadata.review
python -m metadata.review --reject-link <id>     # if needed
python -m metadata.review --approve-all

# 7. Publish: Titan → pgvector, approved links → Neptune
python -m metadata.publish
pytest tests/test_stage5_metadata.py -v      # GATE 2: 19 passed

# 8. Test the agent, then start the server
pytest tests/test_stage6_agent.py -v         # 18 passed
uvicorn api.main:app --port 8000
```

**`python -m` matters** for anything that imports other project files. Run every command from the project folder with `(.venv)` active.

**Re-running is safe.** `apply_schema.py` rebuilds from empty, the generator reproduces identical data from the seed, and `publish` rebuilds pgvector and the graph from the approved items. **Restart the server after publishing**, because notes and graph edges are read once at startup.

## 15. Using it

### Demo page: http://localhost:8000

Type a question, or click an example. Four panels fill in:

- **How it answered:** each of the 11 steps, what it did, and its time
- **Join path:** all 20 tables; the ones this answer used light up; **dashed lines** are the hidden links
- **Answer:** the commentary, what was understood and matched, and the result table
- **SQL and safety checks:** the exact query and a badge per check

Follow-up questions build on the previous one. **New conversation** starts fresh.

### Schema graph: http://localhost:8000/graph

Click a table to light up everything joined to it. The edges come from Neptune, so this page is a view of the graph database itself.

### The API: http://localhost:8000/docs

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which distributor has the highest transaction rejection rate?"}' \
  | python3 -m json.tool
```

| Endpoint | Purpose |
|---|---|
| `POST /query` | Ask a question → `status`, `tables`, `commentary`, `details` (SQL, checks, joins, steps) |
| `GET /schema-graph` | Tables with descriptions, and the join edges from Neptune |
| `GET /graph`, `GET /` | The two pages |
| `GET /health` | `{"status": "ok"}` |

### Looking inside the graph

```python
from adapters import graph_neptune as g

g.counts()          # {'tables': 20, 'joins': 22}
g.run("MATCH (a:Table)-[j:JOINS]->(b:Table) WHERE j.source <> 'declared' "
      "RETURN a.name AS f, j.from_column AS c, b.name AS t")
```

That last query lists the three links the pipeline discovered by itself.

## 16. Tests and gates

| File | Checks | Count |
|---|---|---:|
| `test_rds_setup.py` | Both databases reachable, on one instance; pgvector works; PostgreSQL 15+ | 7 |
| `test_bedrock.py` | `.env` complete; region matches the ARN; chat model returns JSON; Titan returns 1,024 numbers | 5 |
| `test_guardrail.py` | Guardrail reachable; normal question allowed; attack flagged | 3 |
| `test_neptune.py` | Cluster answers; a node can be written, read and deleted; parameters work | 4 |
| `test_stage1_schema.py` | 20 tables, 60 partitions, hidden links undeclared, declared links present | 11 |
| `test_stage2_data.py` | **Gate 1:** no orphans, no negative units, AUM routes agree, all 6 patterns visible | 13 |
| `test_stage5_metadata.py` | **Gate 2:** notes approved, PII flagged, ≥2 of 3 hidden links found, graph correct, search works | 19 |
| `test_stage6_agent.py` | SQL checks refuse unsafe SQL; every demo question answered; attacks and deletes blocked | 18 |
| | **Total** | **80** |

```bash
pytest tests/ -v                                   # everything
pytest tests/test_stage6_agent.py -v -k checks     # fast offline checks only
```

**Gate 1:** nothing moves on until the data is proven correct. **Gate 2:** the agent isn't trusted until the notes and links are scored against the answer key.

## 17. Configuration reference

| File | Setting | Default | Effect |
|---|---|---|---|
| `.env` | `AWS_MODEL_BEDROCK_ID` | — | The chat model for both roles |
| | `AWS_BEDROCK_ARN` | — | The region is read from it |
| | `NEPTUNE_CLUSTER_HOST` / `NEPTUNE_PORT` | — / `8182` | The graph |
| `models.yaml` | `embedding` / `embedding_dimensions` | `amazon.titan-embed-text-v2:0` / `1024` | Must match `vector(1024)` in `metadata/store.py` |
| | `max_tokens` | `4096` | Longest reply allowed |
| `agent.yaml` | `max_sql_retries` | `2` | SQL rewrites after a failed check or run |
| | `run.timeout_ms` / `row_cap` / `max_plan_cost` | `15000` / `5000` / `2000000` | Database protection limits |
| | `entity_min_score` | `0.5` | Name matches below this are ignored |
| | `input_guard.fail_open` | `true` | **Set `false` in production:** refuse when the guardrail can't be reached |
| `metadata.yaml` | `link_discovery.min_overlap` | `0.95` | Share of values that must match to propose a link |
| | `notes.parallel_calls` | `4` | Tables drafted at once. Lower it if throttled |
| | `embedding.max_distinct_values` | `10000` | Columns above this aren't embedded |
| | `graph.max_hops` | `6` | Longest join route searched |
| `generator.yaml` | `scale` / `seed` | `0.2` / `42` | Data size; same seed → identical data |
| `metrics.yaml` | 8 metrics | — | Approved definitions the SQL model must follow |
| `pii.yaml` | `pii_columns` | — | Never sent to Bedrock, never embedded |
| `retrieval.py` | `VECTOR_WEIGHT` | `0.6` | How much of a name match comes from meaning rather than exact words |

**Changing the model:** two lines in `.env`, the ID and the ARN. No code change.

**Changing the embedding size:** Titan V2 can also return 256 or 512 numbers. Change `embedding_dimensions`, change `vector(1024)` in `metadata/store.py`, run `DROP TABLE meta_embeddings;` in `agent_meta`, then re-run build → publish.

## 18. Cost, and keeping it low

| Piece | Idle | In use |
|---|---|---|
| **Amazon Neptune** | Bills by the hour while running, the largest item here | 30-day free trial of the smallest instance for accounts new to Neptune |
| **Amazon RDS** `db.t4g.micro` | Bills by the hour, or free-tier | 20 GB storage within the free allowance |
| **Bedrock** | Nothing | Per token. A full pipeline run is a few dollars; one question is a cent or two |
| **Bedrock Guardrails** | Nothing | A small charge per text checked |
| **IAM, CloudShell, Budgets** | Free | Free |

**Stop both databases when you finish for the day.** This is the single biggest lever:

```bash
aws rds stop-db-instance --region ap-south-1 --db-instance-identifier fundgraph-db
aws neptune stop-db-cluster --region ap-south-1 --db-cluster-identifier fundgraph-graph
```

Start them again with `start-db-instance` and `start-db-cluster`. **Both restart by themselves after 7 days**, so check back.

**Avoid a NAT gateway.** It bills by the hour with no free allowance and is not needed in this design.

## 19. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'boto3'` | Virtual environment not active | `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'metadata'` | Wrong folder, or `-m` left out | `cd` to the project; use `python -m ...` |
| `ConnectionTimeout` on RDS or Neptune | **Your public IP changed**, or the database is stopped | `./scripts/allow_my_ip.sh`, or start it |
| `password authentication failed` | `.env` doesn't match the database password | Update **both** URLs in `.env` |
| `Missing in .env: ...` | A blank setting | Fill it in; compare with `.env.example` |
| `AWS_BEDROCK_ARN does not look like a Bedrock ARN` | A model ID was pasted into the ARN setting | Paste the full ARN |
| `UnrecognizedClientException` | Wrong key pair | Re-copy both keys |
| `INVALID_PAYMENT_INSTRUMENT` | Marketplace can't charge the account | Add a valid default card, or use Amazon's own models |
| `on-demand throughput isn't supported` | A plain model ID was used | Use the inference-profile ID |
| `AccessDenied` on a Neptune query | `neptune-db` permissions missing, or IAM auth off | Apply the Neptune policy; check the cluster setting |
| `ThrottlingException` | Too many requests at once | Retries are automatic; lower `notes.parallel_calls` |
| `No JSON object in model reply` | The model replied in prose | Re-run; if it repeats, raise `max_tokens` |
| `malformed array literal` | The model returned a string where a list was expected | `build.py` handles both; make sure you have the current version |
| `expected 1536 dimensions, not 1024` | Table left from another build | `DROP TABLE meta_embeddings;`, then build → publish |
| New notes not used | Notes are cached at server start | Restart the server |
| A test that passed yesterday fails today | Almost always the IP or a stopped database | Run `test_rds_setup.py` and `test_neptune.py` first |

## 20. Using it on your own database

The pipeline has no table names written into it. To point it at a real database:

1. **Delete `dataplane/`.** You have real data.
2. **Give the agent a read-only user**, ideally on a **read replica**, and set `DATA_DB_URL`.
3. **Keep `agent_meta` on your own small RDS instance**, with pgvector, and set `AGENT_DB_URL`.
4. **List your personal-data columns** in `config/pii.yaml`.
5. **Write your business definitions** in `config/metrics.yaml`, and have the business team sign them off.
6. **Run build → review → publish.** Read every drafted note and proposed link before approving: this review is what makes the answers trustworthy.
7. **Write your own questions** in `evals/questions/questions.yaml` with known correct answers, and run the evaluation to get a baseline.

**If your database isn't PostgreSQL:** write one new adapter with the same four functions as `adapters/postgres.py` (read catalog, read statistics, plan estimate, run read-only). Nothing else changes.

**If Neptune is more than you need:** the graph is 20 nodes and 22 edges, and `meta_edges` in PostgreSQL is already the source of truth. A recursive SQL query over that table would find the same routes with no extra service. Neptune earns its place when the client's schema is large, when a real graph database is wanted for its own sake, or when the graph will be queried directly. Swapping it out is one file: `metadata/retrieval.py` takes a list of edges, wherever they come from.
