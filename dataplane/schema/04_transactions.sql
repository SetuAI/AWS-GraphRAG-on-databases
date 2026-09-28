-- =============================================================================
-- dataplane/schema/04_transactions.sql
-- =============================================================================
--
-- WHAT THIS FILE IS FOR
-- ---------------------
-- Creates the transactions table: every purchase, redemption, SIP instalment
-- and switch made by every investor over 5 years (September 2021 to
-- August 2026). About 4.6 million rows at scale 1.
--
-- It is the biggest table by far, which is why it gets a file of its own and
-- is PARTITIONED (split into monthly pieces). See "WHY PARTITION" below.
--
-- WHERE THIS FITS
-- ---------------
-- File 4 of 4. Must run last, because transactions points to folios and
-- scheme_plans, created in files 03 and 01.
--
-- ON AWS
-- ------
-- This runs in mf_data on Amazon RDS for PostgreSQL. Partitioning matters more
-- here than it did locally: on RDS you pay for the input/output a query causes,
-- so a query that reads all 60 months costs more than one that reads a single
-- month. That is why the Stage 6 guardrail refuses any query on this table
-- without a real date range.
--
-- Most business questions end up here. "Net flows" (money in minus money
-- out), "redemptions", "rejection rate" and "SIP book" are all calculated
-- from this table.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- CLEAN START (same reasoning as file 01)
-- -----------------------------------------------------------------------------
-- Dropping the main table also drops all 60 monthly pieces attached to it.
DROP TABLE IF EXISTS transactions CASCADE;


-- =============================================================================
-- WHY PARTITION
-- =============================================================================
-- Without partitioning, a question about March 2024 makes Postgres scan all
-- 4.6 million rows (79 million at scale 15) to find the ones from March 2024.
--
-- With partitioning, the rows are stored in 60 separate pieces, one per
-- month. When a query filters on txn_date, Postgres reads only the pieces
-- for the months asked about and skips the rest. This is called
-- "partition pruning".
--
-- This is also why the guardrails in Stage 7 REQUIRE a date filter on any
-- query against transactions: without one, pruning cannot happen and the
-- query reads everything.
-- =============================================================================

-- "PARTITION BY RANGE (txn_date)" makes this a parent table. It stores no
-- rows itself; every row is sent to the monthly piece that covers its date.
CREATE TABLE transactions (
    transaction_id      bigint NOT NULL,

    -- The business date of the transaction. The partition key: this value
    -- decides which monthly piece a row is stored in.
    txn_date            date NOT NULL,

    folio_id            bigint NOT NULL REFERENCES folios (folio_id),
    plan_id             integer NOT NULL REFERENCES scheme_plans (plan_id),

    -- Money in:  Purchase, SIP, Switch In, Dividend Reinvestment
    -- Money out: Redemption, Switch Out
    txn_type            text NOT NULL
        CHECK (txn_type IN ('Purchase', 'Redemption', 'SIP',
                            'Switch In', 'Switch Out', 'Dividend Reinvestment')),

    -- Amount in rupees, always positive. Whether it is money in or out is
    -- decided by txn_type, never by a minus sign. This avoids double
    -- negatives in net-flow calculations.
    amount              numeric(18, 2) NOT NULL CHECK (amount > 0),

    -- Units bought or sold, and the NAV they were priced at.
    -- The Stage 3 validation checks that units x nav = amount, within Rs 1.
    units               numeric(18, 3) NOT NULL CHECK (units > 0),
    nav                 numeric(12, 4) NOT NULL CHECK (nav > 0),

    txn_status          text NOT NULL
        CHECK (txn_status IN ('Completed', 'Rejected', 'Pending')),

    -- Why a transaction was rejected. Filled only when txn_status = 'Rejected'.
    rejection_reason    text,

    -- HIDDEN LINK 3 OF 3
    -- arn_code holds the same values as distributors.arn_no, but there is no
    -- REFERENCES and the column names differ. Stage 5 can find this only by
    -- checking that the values overlap.
    -- NULL means a direct transaction, with no distributor involved.
    arn_code            text,                -- hidden link: no REFERENCES on purpose

    channel             text NOT NULL
        CHECK (channel IN ('Online', 'Branch', 'Distributor', 'Mobile App')),

    created_at          timestamptz NOT NULL DEFAULT now(),

    -- A rejection reason is required when rejected, and not allowed otherwise.
    CHECK ((txn_status = 'Rejected') = (rejection_reason IS NOT NULL)),

    -- On a partitioned table, the primary key MUST include the partition key
    -- (txn_date). Postgres can only guarantee uniqueness within each monthly
    -- piece, so txn_date must be part of the key.
    PRIMARY KEY (transaction_id, txn_date)
) PARTITION BY RANGE (txn_date);


-- =============================================================================
-- CREATE THE 60 MONTHLY PIECES
-- =============================================================================
-- Writing 60 CREATE TABLE statements by hand would be long and easy to get
-- wrong. This block loops over the months and creates one piece per month.
--
-- DO $$ ... $$  runs a small program inside Postgres (in its PL/pgSQL language).
DO $$
DECLARE
    -- The first day of the month we are creating a piece for.
    month_start date;
BEGIN
    -- generate_series gives one value per month, from 1 Sep 2021 to 1 Aug 2026.
    FOR month_start IN
        SELECT generate_series(date '2021-09-01', date '2026-08-01', interval '1 month')::date
    LOOP
        -- format() builds the SQL text, then EXECUTE runs it.
        --   %I -> inserted as a table name, e.g. transactions_2024_03
        --   %L -> inserted as a quoted value, e.g. '2024-03-01'
        --
        -- FROM (start) TO (end): "start" is included, "end" is NOT included.
        -- So FROM 2024-03-01 TO 2024-04-01 holds every date in March 2024,
        -- and 1 April goes into the April piece.
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF transactions FOR VALUES FROM (%L) TO (%L)',
            'transactions_' || to_char(month_start, 'YYYY_MM'),
            month_start,
            (month_start + interval '1 month')::date
        );
    END LOOP;
END
$$;

-- There is deliberately NO "default" piece for dates outside these 60 months.
-- If the generator ever produces a date outside the 5-year window, the insert
-- fails loudly, instead of the row quietly landing somewhere unexpected.


-- =============================================================================
-- INDEXES
-- =============================================================================
-- Created on the parent table, Postgres automatically creates the matching
-- index on every monthly piece.

-- "All transactions in this folio": used by investor-level questions.
CREATE INDEX transactions_folio_idx ON transactions (folio_id);

-- "All transactions for this plan in this date range": used by net-flow
-- and redemption questions, the most common kind.
CREATE INDEX transactions_plan_date_idx ON transactions (plan_id, txn_date);

-- "All transactions through this distributor": used by the rejection-rate question.
CREATE INDEX transactions_arn_idx ON transactions (arn_code);