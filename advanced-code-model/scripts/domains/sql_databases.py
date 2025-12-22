"""
SQL & Databases Domain

Covers all major SQL dialects and database systems:
- PostgreSQL
- MySQL
- Oracle
- SQLite
- Amazon Redshift
- Snowflake
- SparkSQL
- And common patterns across all
"""

from typing import List, Optional
from .base import BaseDomain, DomainExample


class SQLDatabasesDomain(BaseDomain):
    """SQL and database training examples across multiple flavors."""

    def __init__(self, flavor: Optional[str] = None):
        super().__init__()
        self.flavor = flavor  # None means all flavors

    def get_name(self) -> str:
        if self.flavor:
            return f"SQL ({self.flavor.upper()})"
        return "SQL & Databases"

    def get_description(self) -> str:
        return "SQL queries, database design, and data manipulation across PostgreSQL, MySQL, Oracle, SQLite, Redshift, Snowflake, and SparkSQL"

    def get_subdomains(self) -> List[str]:
        return [
            "postgres", "mysql", "oracle", "sqlite",
            "redshift", "snowflake", "sparksql",
            "ddl", "dml", "analytics", "optimization"
        ]

    def get_examples(self) -> List[DomainExample]:
        examples = []

        # Add examples for each flavor
        if not self.flavor or self.flavor == "postgres":
            examples.extend(self._postgres_examples())
        if not self.flavor or self.flavor == "mysql":
            examples.extend(self._mysql_examples())
        if not self.flavor or self.flavor == "oracle":
            examples.extend(self._oracle_examples())
        if not self.flavor or self.flavor == "sqlite":
            examples.extend(self._sqlite_examples())
        if not self.flavor or self.flavor == "redshift":
            examples.extend(self._redshift_examples())
        if not self.flavor or self.flavor == "snowflake":
            examples.extend(self._snowflake_examples())
        if not self.flavor or self.flavor == "sparksql":
            examples.extend(self._sparksql_examples())

        # Common patterns
        examples.extend(self._common_patterns())

        return examples

    def _postgres_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a PostgreSQL table for users with UUID primary key and JSONB metadata",
                code="""-- PostgreSQL: Users table with UUID and JSONB
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Index for JSONB queries
CREATE INDEX idx_users_metadata ON users USING GIN (metadata);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();""",
                domain="sql",
                subdomain="postgres",
                tags=["ddl", "uuid", "jsonb", "trigger"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Write a PostgreSQL query with window functions to calculate running totals and rankings",
                code="""-- PostgreSQL: Window functions for analytics
WITH sales_data AS (
    SELECT
        salesperson_id,
        sale_date,
        amount,
        -- Running total per salesperson
        SUM(amount) OVER (
            PARTITION BY salesperson_id
            ORDER BY sale_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_total,
        -- Rank within each month
        RANK() OVER (
            PARTITION BY DATE_TRUNC('month', sale_date)
            ORDER BY amount DESC
        ) AS monthly_rank,
        -- Percentile
        PERCENT_RANK() OVER (
            PARTITION BY salesperson_id
            ORDER BY amount
        ) AS percentile,
        -- Moving average (last 7 days)
        AVG(amount) OVER (
            PARTITION BY salesperson_id
            ORDER BY sale_date
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS moving_avg_7d
    FROM sales
)
SELECT *
FROM sales_data
WHERE monthly_rank <= 10
ORDER BY sale_date DESC, monthly_rank;""",
                domain="sql",
                subdomain="postgres",
                tags=["window_functions", "analytics", "cte"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create PostgreSQL full-text search with tsvector and tsquery",
                code="""-- PostgreSQL: Full-text search setup
-- Add tsvector column
ALTER TABLE articles ADD COLUMN search_vector tsvector;

-- Update search vector with weighted content
UPDATE articles SET search_vector =
    setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(summary, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(content, '')), 'C');

-- Create GIN index for fast search
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- Trigger to auto-update search vector
CREATE OR REPLACE FUNCTION articles_search_trigger()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.summary, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER articles_search_update
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION articles_search_trigger();

-- Search query with ranking
SELECT
    id,
    title,
    ts_rank(search_vector, query) AS rank,
    ts_headline('english', content, query, 'MaxWords=50') AS snippet
FROM articles,
     to_tsquery('english', 'python & machine & learning') AS query
WHERE search_vector @@ query
ORDER BY rank DESC
LIMIT 20;""",
                domain="sql",
                subdomain="postgres",
                tags=["full_text_search", "tsvector", "gin_index"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create PostgreSQL partitioned table for time-series data",
                code="""-- PostgreSQL: Range partitioning for time-series
CREATE TABLE events (
    id BIGSERIAL,
    event_type VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create partitions for each month
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE events_2024_02 PARTITION OF events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE events_2024_03 PARTITION OF events
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- Default partition for future data
CREATE TABLE events_default PARTITION OF events DEFAULT;

-- Create indexes on partitions
CREATE INDEX idx_events_user_id ON events (user_id);
CREATE INDEX idx_events_type ON events (event_type);

-- Function to auto-create partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(target_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    start_date := DATE_TRUNC('month', target_date);
    end_date := start_date + INTERVAL '1 month';
    partition_name := 'events_' || TO_CHAR(start_date, 'YYYY_MM');

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF events
         FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
END;
$$ LANGUAGE plpgsql;""",
                domain="sql",
                subdomain="postgres",
                tags=["partitioning", "time_series", "performance"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Write PostgreSQL recursive CTE for hierarchical org chart",
                code="""-- PostgreSQL: Recursive CTE for org hierarchy
WITH RECURSIVE org_hierarchy AS (
    -- Base case: top-level managers (no manager)
    SELECT
        id,
        name,
        title,
        manager_id,
        1 AS level,
        ARRAY[id] AS path,
        name AS hierarchy_path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees with managers
    SELECT
        e.id,
        e.name,
        e.title,
        e.manager_id,
        oh.level + 1,
        oh.path || e.id,
        oh.hierarchy_path || ' > ' || e.name
    FROM employees e
    INNER JOIN org_hierarchy oh ON e.manager_id = oh.id
    WHERE NOT e.id = ANY(oh.path)  -- Prevent cycles
)
SELECT
    id,
    REPEAT('  ', level - 1) || name AS org_tree,
    title,
    level,
    hierarchy_path,
    (SELECT COUNT(*) FROM org_hierarchy sub WHERE id = ANY(sub.path) AND sub.id != id) AS team_size
FROM org_hierarchy
ORDER BY path;""",
                domain="sql",
                subdomain="postgres",
                tags=["recursive_cte", "hierarchy", "tree"],
                difficulty="advanced"
            ),
        ]

    def _mysql_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create MySQL table with proper indexing for e-commerce products",
                code="""-- MySQL: E-commerce products table
CREATE TABLE products (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    sku VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    compare_at_price DECIMAL(10, 2),
    cost DECIMAL(10, 2),
    quantity INT UNSIGNED DEFAULT 0,
    category_id INT UNSIGNED,
    brand_id INT UNSIGNED,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_category (category_id),
    INDEX idx_brand (brand_id),
    INDEX idx_price (price),
    INDEX idx_active_category (is_active, category_id),
    FULLTEXT INDEX idx_search (name, description),

    -- Foreign keys
    CONSTRAINT fk_category FOREIGN KEY (category_id)
        REFERENCES categories(id) ON DELETE SET NULL,
    CONSTRAINT fk_brand FOREIGN KEY (brand_id)
        REFERENCES brands(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Stored procedure for inventory update
DELIMITER //
CREATE PROCEDURE update_inventory(
    IN p_sku VARCHAR(50),
    IN p_quantity_change INT,
    OUT p_new_quantity INT
)
BEGIN
    DECLARE current_qty INT;

    START TRANSACTION;

    SELECT quantity INTO current_qty
    FROM products
    WHERE sku = p_sku
    FOR UPDATE;

    IF current_qty + p_quantity_change < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Insufficient inventory';
    END IF;

    UPDATE products
    SET quantity = quantity + p_quantity_change
    WHERE sku = p_sku;

    SELECT quantity INTO p_new_quantity
    FROM products WHERE sku = p_sku;

    COMMIT;
END //
DELIMITER ;""",
                domain="sql",
                subdomain="mysql",
                tags=["ddl", "stored_procedure", "transaction"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Write MySQL query for sales analytics with date functions",
                code="""-- MySQL: Sales analytics dashboard query
SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(quantity) AS items_sold,
    ROUND(SUM(total_amount), 2) AS revenue,
    ROUND(AVG(total_amount), 2) AS avg_order_value,
    ROUND(SUM(total_amount) / COUNT(DISTINCT customer_id), 2) AS revenue_per_customer,

    -- Year-over-year comparison
    LAG(SUM(total_amount), 12) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')) AS revenue_last_year,
    ROUND(
        (SUM(total_amount) - LAG(SUM(total_amount), 12) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')))
        / NULLIF(LAG(SUM(total_amount), 12) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')), 0) * 100,
        2
    ) AS yoy_growth_pct,

    -- Month-over-month
    LAG(SUM(total_amount), 1) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')) AS revenue_last_month,
    ROUND(
        (SUM(total_amount) - LAG(SUM(total_amount), 1) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')))
        / NULLIF(LAG(SUM(total_amount), 1) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')), 0) * 100,
        2
    ) AS mom_growth_pct

FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR)
    AND order_status = 'completed'
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month DESC;""",
                domain="sql",
                subdomain="mysql",
                tags=["analytics", "window_functions", "date_functions"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create MySQL event scheduler for data cleanup",
                code="""-- MySQL: Event scheduler for maintenance tasks
-- Enable event scheduler
SET GLOBAL event_scheduler = ON;

-- Archive old orders (older than 2 years)
DELIMITER //
CREATE EVENT IF NOT EXISTS archive_old_orders
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_DATE + INTERVAL 1 DAY + INTERVAL 2 HOUR
DO
BEGIN
    DECLARE batch_size INT DEFAULT 1000;
    DECLARE rows_moved INT DEFAULT 0;

    -- Move to archive table in batches
    REPEAT
        INSERT INTO orders_archive
        SELECT * FROM orders
        WHERE order_date < DATE_SUB(CURDATE(), INTERVAL 2 YEAR)
            AND archived_at IS NULL
        LIMIT batch_size;

        SET rows_moved = ROW_COUNT();

        UPDATE orders
        SET archived_at = NOW()
        WHERE order_date < DATE_SUB(CURDATE(), INTERVAL 2 YEAR)
            AND archived_at IS NULL
        LIMIT batch_size;

        -- Small delay to reduce lock contention
        DO SLEEP(0.1);

    UNTIL rows_moved < batch_size
    END REPEAT;

    -- Log completion
    INSERT INTO maintenance_log (task_name, completed_at, rows_affected)
    VALUES ('archive_old_orders', NOW(), rows_moved);
END //

-- Cleanup expired sessions
CREATE EVENT IF NOT EXISTS cleanup_expired_sessions
ON SCHEDULE EVERY 1 HOUR
DO
BEGIN
    DELETE FROM user_sessions
    WHERE expires_at < NOW()
    LIMIT 10000;

    INSERT INTO maintenance_log (task_name, completed_at, rows_affected)
    VALUES ('cleanup_expired_sessions', NOW(), ROW_COUNT());
END //

-- Update statistics table
CREATE EVENT IF NOT EXISTS update_daily_stats
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_DATE + INTERVAL 1 DAY
DO
BEGIN
    INSERT INTO daily_stats (stat_date, total_orders, total_revenue, new_customers)
    SELECT
        CURDATE() - INTERVAL 1 DAY,
        COUNT(DISTINCT order_id),
        SUM(total_amount),
        (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = CURDATE() - INTERVAL 1 DAY)
    FROM orders
    WHERE DATE(order_date) = CURDATE() - INTERVAL 1 DAY;
END //
DELIMITER ;""",
                domain="sql",
                subdomain="mysql",
                tags=["events", "maintenance", "automation"],
                difficulty="advanced"
            ),
        ]

    def _oracle_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create Oracle PL/SQL package for order processing",
                code="""-- Oracle: PL/SQL package for order processing
CREATE OR REPLACE PACKAGE order_pkg AS
    -- Types
    TYPE order_item_rec IS RECORD (
        product_id NUMBER,
        quantity NUMBER,
        unit_price NUMBER(10,2)
    );
    TYPE order_items_tab IS TABLE OF order_item_rec;

    -- Exceptions
    insufficient_stock EXCEPTION;
    invalid_customer EXCEPTION;
    PRAGMA EXCEPTION_INIT(insufficient_stock, -20001);
    PRAGMA EXCEPTION_INIT(invalid_customer, -20002);

    -- Procedures
    PROCEDURE create_order(
        p_customer_id IN NUMBER,
        p_items IN order_items_tab,
        p_order_id OUT NUMBER
    );

    PROCEDURE cancel_order(p_order_id IN NUMBER);

    FUNCTION get_order_total(p_order_id IN NUMBER) RETURN NUMBER;
END order_pkg;
/

CREATE OR REPLACE PACKAGE BODY order_pkg AS

    PROCEDURE create_order(
        p_customer_id IN NUMBER,
        p_items IN order_items_tab,
        p_order_id OUT NUMBER
    ) IS
        v_customer_exists NUMBER;
        v_stock NUMBER;
        v_total NUMBER := 0;
    BEGIN
        -- Validate customer
        SELECT COUNT(*) INTO v_customer_exists
        FROM customers WHERE customer_id = p_customer_id;

        IF v_customer_exists = 0 THEN
            RAISE invalid_customer;
        END IF;

        -- Create order header
        SELECT order_seq.NEXTVAL INTO p_order_id FROM DUAL;

        INSERT INTO orders (order_id, customer_id, order_date, status)
        VALUES (p_order_id, p_customer_id, SYSDATE, 'PENDING');

        -- Process items
        FOR i IN 1..p_items.COUNT LOOP
            -- Check stock
            SELECT quantity_available INTO v_stock
            FROM products
            WHERE product_id = p_items(i).product_id
            FOR UPDATE;

            IF v_stock < p_items(i).quantity THEN
                RAISE insufficient_stock;
            END IF;

            -- Insert order item
            INSERT INTO order_items (order_id, product_id, quantity, unit_price)
            VALUES (p_order_id, p_items(i).product_id,
                    p_items(i).quantity, p_items(i).unit_price);

            -- Update stock
            UPDATE products
            SET quantity_available = quantity_available - p_items(i).quantity
            WHERE product_id = p_items(i).product_id;

            v_total := v_total + (p_items(i).quantity * p_items(i).unit_price);
        END LOOP;

        -- Update order total
        UPDATE orders SET total_amount = v_total WHERE order_id = p_order_id;

        COMMIT;

    EXCEPTION
        WHEN insufficient_stock THEN
            ROLLBACK;
            RAISE_APPLICATION_ERROR(-20001, 'Insufficient stock for order');
        WHEN invalid_customer THEN
            ROLLBACK;
            RAISE_APPLICATION_ERROR(-20002, 'Invalid customer ID');
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END create_order;

    PROCEDURE cancel_order(p_order_id IN NUMBER) IS
    BEGIN
        -- Restore inventory
        FOR item IN (SELECT product_id, quantity FROM order_items WHERE order_id = p_order_id) LOOP
            UPDATE products
            SET quantity_available = quantity_available + item.quantity
            WHERE product_id = item.product_id;
        END LOOP;

        UPDATE orders SET status = 'CANCELLED' WHERE order_id = p_order_id;
        COMMIT;
    END cancel_order;

    FUNCTION get_order_total(p_order_id IN NUMBER) RETURN NUMBER IS
        v_total NUMBER;
    BEGIN
        SELECT NVL(SUM(quantity * unit_price), 0) INTO v_total
        FROM order_items
        WHERE order_id = p_order_id;
        RETURN v_total;
    END get_order_total;

END order_pkg;
/""",
                domain="sql",
                subdomain="oracle",
                tags=["plsql", "package", "transaction"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Write Oracle analytical query with MODEL clause",
                code="""-- Oracle: Advanced analytics with MODEL clause
SELECT *
FROM (
    SELECT
        product_category,
        sale_month,
        revenue,
        units_sold
    FROM monthly_sales
    WHERE sale_year = 2024
)
MODEL
    PARTITION BY (product_category)
    DIMENSION BY (sale_month)
    MEASURES (revenue, units_sold, 0 AS forecast_revenue, 0 AS growth_rate)
    RULES (
        -- Calculate month-over-month growth
        growth_rate[ANY] = CASE
            WHEN revenue[CV()-1] > 0 THEN
                ROUND((revenue[CV()] - revenue[CV()-1]) / revenue[CV()-1] * 100, 2)
            ELSE 0
        END,

        -- Forecast next 3 months using simple moving average
        forecast_revenue[FOR sale_month FROM 13 TO 15 INCREMENT 1] =
            ROUND(AVG(revenue)[sale_month BETWEEN CV()-3 AND CV()-1], 2)
    )
ORDER BY product_category, sale_month;

-- Alternative with MATCH_RECOGNIZE for pattern matching
SELECT *
FROM stock_prices
MATCH_RECOGNIZE (
    PARTITION BY symbol
    ORDER BY trade_date
    MEASURES
        FIRST(trade_date) AS pattern_start,
        LAST(trade_date) AS pattern_end,
        FIRST(close_price) AS start_price,
        LAST(close_price) AS end_price,
        COUNT(*) AS pattern_length
    ONE ROW PER MATCH
    PATTERN (STRT DOWN+ UP+)
    DEFINE
        DOWN AS close_price < PREV(close_price),
        UP AS close_price > PREV(close_price)
);""",
                domain="sql",
                subdomain="oracle",
                tags=["model_clause", "analytics", "pattern_matching"],
                difficulty="advanced"
            ),
        ]

    def _sqlite_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create SQLite database schema for a note-taking app",
                code="""-- SQLite: Note-taking app schema
-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    last_login TEXT
);

-- Notebooks/folders
CREATE TABLE notebooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    color TEXT DEFAULT '#3498db',
    parent_id INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES notebooks(id) ON DELETE CASCADE
);

-- Notes with FTS5 for full-text search
CREATE TABLE notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    notebook_id INTEGER,
    title TEXT NOT NULL,
    content TEXT,
    is_pinned INTEGER DEFAULT 0,
    is_archived INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE SET NULL
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE notes_fts USING fts5(
    title,
    content,
    content='notes',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content)
    VALUES (NEW.id, NEW.title, NEW.content);
END;

CREATE TRIGGER notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
    VALUES ('delete', OLD.id, OLD.title, OLD.content);
END;

CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
    VALUES ('delete', OLD.id, OLD.title, OLD.content);
    INSERT INTO notes_fts(rowid, title, content)
    VALUES (NEW.id, NEW.title, NEW.content);
END;

-- Tags
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    UNIQUE(user_id, name),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE note_tags (
    note_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (note_id, tag_id),
    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_notes_user ON notes(user_id);
CREATE INDEX idx_notes_notebook ON notes(notebook_id);
CREATE INDEX idx_notes_updated ON notes(updated_at DESC);

-- Search query example
-- SELECT n.*, snippet(notes_fts, 1, '<mark>', '</mark>', '...', 32) as snippet
-- FROM notes n
-- JOIN notes_fts fts ON n.id = fts.rowid
-- WHERE notes_fts MATCH 'python OR programming'
-- ORDER BY rank;""",
                domain="sql",
                subdomain="sqlite",
                tags=["fts5", "triggers", "schema_design"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Write SQLite queries for analytics with CTEs and window functions",
                code="""-- SQLite: Analytics queries
-- Note: SQLite 3.25+ supports window functions

-- Daily active users with 7-day rolling average
WITH daily_stats AS (
    SELECT
        date(created_at) as day,
        COUNT(DISTINCT user_id) as dau
    FROM user_actions
    GROUP BY date(created_at)
)
SELECT
    day,
    dau,
    ROUND(AVG(dau) OVER (
        ORDER BY day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) as rolling_7d_avg,
    dau - LAG(dau, 7) OVER (ORDER BY day) as wow_change
FROM daily_stats
ORDER BY day DESC
LIMIT 30;

-- Cohort retention analysis
WITH user_cohorts AS (
    SELECT
        user_id,
        date(created_at) as cohort_date,
        strftime('%Y-%W', created_at) as cohort_week
    FROM users
),
user_activity AS (
    SELECT DISTINCT
        user_id,
        date(activity_date) as activity_date,
        strftime('%Y-%W', activity_date) as activity_week
    FROM user_actions
),
retention AS (
    SELECT
        uc.cohort_week,
        COUNT(DISTINCT uc.user_id) as cohort_size,
        SUM(CASE WHEN ua.activity_week = uc.cohort_week THEN 1 ELSE 0 END) as week_0,
        SUM(CASE WHEN julianday(ua.activity_date) - julianday(uc.cohort_date) BETWEEN 7 AND 13 THEN 1 ELSE 0 END) as week_1,
        SUM(CASE WHEN julianday(ua.activity_date) - julianday(uc.cohort_date) BETWEEN 14 AND 20 THEN 1 ELSE 0 END) as week_2,
        SUM(CASE WHEN julianday(ua.activity_date) - julianday(uc.cohort_date) BETWEEN 21 AND 27 THEN 1 ELSE 0 END) as week_3
    FROM user_cohorts uc
    LEFT JOIN user_activity ua ON uc.user_id = ua.user_id
    GROUP BY uc.cohort_week
)
SELECT
    cohort_week,
    cohort_size,
    ROUND(100.0 * week_0 / cohort_size, 1) as 'Week 0 %',
    ROUND(100.0 * week_1 / cohort_size, 1) as 'Week 1 %',
    ROUND(100.0 * week_2 / cohort_size, 1) as 'Week 2 %',
    ROUND(100.0 * week_3 / cohort_size, 1) as 'Week 3 %'
FROM retention
ORDER BY cohort_week DESC;""",
                domain="sql",
                subdomain="sqlite",
                tags=["analytics", "window_functions", "cohort"],
                difficulty="advanced"
            ),
        ]

    def _redshift_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create Redshift table with distribution and sort keys for analytics",
                code="""-- Amazon Redshift: Optimized analytics table
-- Events table with proper distribution
CREATE TABLE events (
    event_id BIGINT IDENTITY(1,1),
    event_timestamp TIMESTAMP NOT NULL ENCODE az64,
    event_type VARCHAR(50) NOT NULL ENCODE lzo,
    user_id BIGINT NOT NULL ENCODE az64,
    session_id VARCHAR(64) ENCODE lzo,
    page_url VARCHAR(2048) ENCODE lzo,
    referrer_url VARCHAR(2048) ENCODE lzo,
    device_type VARCHAR(20) ENCODE bytedict,
    country_code CHAR(2) ENCODE bytedict,
    properties SUPER,  -- Semi-structured data
    created_date DATE NOT NULL ENCODE az64
)
DISTKEY(user_id)
SORTKEY(created_date, event_timestamp)
DISTSTYLE KEY;

-- Users dimension table (all distribution for small tables)
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY ENCODE az64,
    email VARCHAR(255) ENCODE lzo,
    signup_date DATE ENCODE az64,
    subscription_tier VARCHAR(20) ENCODE bytedict,
    lifetime_value DECIMAL(12,2) ENCODE az64
)
DISTSTYLE ALL;

-- Materialized view for daily aggregations
CREATE MATERIALIZED VIEW daily_event_stats
DISTKEY(event_date)
SORTKEY(event_date, event_type)
AUTO REFRESH YES
AS
SELECT
    DATE(event_timestamp) as event_date,
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT session_id) as unique_sessions
FROM events
GROUP BY 1, 2;

-- Query using SUPER type
SELECT
    event_type,
    properties.button_name::VARCHAR as button,
    properties.value::INT as value,
    COUNT(*) as clicks
FROM events
WHERE event_type = 'button_click'
    AND properties.button_name IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY clicks DESC;""",
                domain="sql",
                subdomain="redshift",
                tags=["distribution", "sort_keys", "super_type"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Write Redshift query with Spectrum for S3 data lake queries",
                code="""-- Redshift Spectrum: Query S3 data lake
-- Create external schema pointing to Glue catalog
CREATE EXTERNAL SCHEMA spectrum_schema
FROM DATA CATALOG
DATABASE 'datalake'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftSpectrumRole'
CREATE EXTERNAL DATABASE IF NOT EXISTS;

-- External table for parquet files in S3
CREATE EXTERNAL TABLE spectrum_schema.raw_events (
    event_id BIGINT,
    event_type VARCHAR(50),
    user_id BIGINT,
    properties VARCHAR(MAX),
    event_timestamp TIMESTAMP
)
PARTITIONED BY (year INT, month INT, day INT)
STORED AS PARQUET
LOCATION 's3://my-datalake/events/';

-- Add partitions
ALTER TABLE spectrum_schema.raw_events
ADD PARTITION (year=2024, month=1, day=15)
LOCATION 's3://my-datalake/events/year=2024/month=01/day=15/';

-- Query combining Redshift and Spectrum data
WITH spectrum_events AS (
    SELECT
        user_id,
        event_type,
        COUNT(*) as event_count
    FROM spectrum_schema.raw_events
    WHERE year = 2024 AND month = 1
    GROUP BY 1, 2
),
redshift_users AS (
    SELECT user_id, email, subscription_tier
    FROM users
    WHERE subscription_tier = 'premium'
)
SELECT
    u.email,
    u.subscription_tier,
    e.event_type,
    e.event_count,
    PERCENT_RANK() OVER (
        PARTITION BY e.event_type
        ORDER BY e.event_count
    ) as percentile
FROM redshift_users u
JOIN spectrum_events e ON u.user_id = e.user_id
ORDER BY e.event_count DESC;

-- Unload results to S3
UNLOAD ('SELECT * FROM daily_event_stats WHERE event_date >= ''2024-01-01''')
TO 's3://my-bucket/exports/daily_stats_'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftS3Role'
FORMAT PARQUET
PARTITION BY (event_date);""",
                domain="sql",
                subdomain="redshift",
                tags=["spectrum", "s3", "data_lake"],
                difficulty="advanced"
            ),
        ]

    def _snowflake_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create Snowflake data pipeline with streams and tasks",
                code="""-- Snowflake: CDC pipeline with streams and tasks

-- Source table
CREATE OR REPLACE TABLE raw_orders (
    order_id NUMBER AUTOINCREMENT,
    customer_id NUMBER,
    product_id NUMBER,
    quantity NUMBER,
    unit_price NUMBER(10,2),
    order_date TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    status VARCHAR(20) DEFAULT 'pending'
);

-- Create stream to capture changes
CREATE OR REPLACE STREAM orders_stream ON TABLE raw_orders
    APPEND_ONLY = FALSE
    SHOW_INITIAL_ROWS = FALSE;

-- Staging table for transformations
CREATE OR REPLACE TABLE orders_enriched (
    order_id NUMBER,
    customer_id NUMBER,
    customer_name VARCHAR,
    product_id NUMBER,
    product_name VARCHAR,
    quantity NUMBER,
    unit_price NUMBER(10,2),
    total_amount NUMBER(10,2),
    order_date TIMESTAMP_NTZ,
    status VARCHAR(20),
    processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Task to process stream data
CREATE OR REPLACE TASK process_orders_task
    WAREHOUSE = 'COMPUTE_WH'
    SCHEDULE = '1 MINUTE'
    WHEN SYSTEM$STREAM_HAS_DATA('orders_stream')
AS
MERGE INTO orders_enriched t
USING (
    SELECT
        s.order_id,
        s.customer_id,
        c.customer_name,
        s.product_id,
        p.product_name,
        s.quantity,
        s.unit_price,
        s.quantity * s.unit_price as total_amount,
        s.order_date,
        s.status
    FROM orders_stream s
    LEFT JOIN customers c ON s.customer_id = c.customer_id
    LEFT JOIN products p ON s.product_id = p.product_id
    WHERE s.METADATA$ACTION = 'INSERT'
) AS source
ON t.order_id = source.order_id
WHEN MATCHED THEN UPDATE SET
    t.status = source.status,
    t.processed_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (
    order_id, customer_id, customer_name, product_id, product_name,
    quantity, unit_price, total_amount, order_date, status
) VALUES (
    source.order_id, source.customer_id, source.customer_name,
    source.product_id, source.product_name, source.quantity,
    source.unit_price, source.total_amount, source.order_date, source.status
);

-- Enable task
ALTER TASK process_orders_task RESUME;

-- Dynamic table for real-time aggregations
CREATE OR REPLACE DYNAMIC TABLE hourly_sales_summary
    TARGET_LAG = '1 hour'
    WAREHOUSE = 'COMPUTE_WH'
AS
SELECT
    DATE_TRUNC('hour', order_date) as hour,
    COUNT(*) as order_count,
    SUM(total_amount) as revenue,
    AVG(total_amount) as avg_order_value,
    COUNT(DISTINCT customer_id) as unique_customers
FROM orders_enriched
WHERE status = 'completed'
GROUP BY 1;""",
                domain="sql",
                subdomain="snowflake",
                tags=["streams", "tasks", "cdc", "dynamic_tables"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Write Snowflake query with semi-structured data and time travel",
                code="""-- Snowflake: Semi-structured data and time travel

-- Table with VARIANT column for JSON
CREATE OR REPLACE TABLE api_events (
    event_id NUMBER AUTOINCREMENT,
    event_time TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    event_data VARIANT
);

-- Insert JSON data
INSERT INTO api_events (event_data)
SELECT PARSE_JSON('{
    "event_type": "page_view",
    "user": {
        "id": 12345,
        "email": "user@example.com",
        "preferences": ["dark_mode", "notifications"]
    },
    "page": {
        "url": "/products/123",
        "title": "Product Details"
    },
    "metadata": {
        "device": "mobile",
        "browser": "chrome",
        "ip": "192.168.1.1"
    }
}');

-- Query semi-structured data
SELECT
    event_id,
    event_time,
    event_data:event_type::STRING as event_type,
    event_data:user.id::NUMBER as user_id,
    event_data:user.email::STRING as email,
    event_data:page.url::STRING as page_url,
    event_data:metadata.device::STRING as device,
    -- Flatten arrays
    pref.value::STRING as preference
FROM api_events,
    LATERAL FLATTEN(input => event_data:user.preferences) pref
WHERE event_data:event_type = 'page_view';

-- Time travel queries
-- Query data as of 1 hour ago
SELECT * FROM orders_enriched
AT(OFFSET => -60*60);

-- Query data as of specific timestamp
SELECT * FROM orders_enriched
AT(TIMESTAMP => '2024-01-15 10:00:00'::TIMESTAMP_NTZ);

-- Query data before specific query ID
SELECT * FROM orders_enriched
BEFORE(STATEMENT => '01a1b2c3-0000-0000-0000-000000000000');

-- Clone table at point in time
CREATE TABLE orders_backup CLONE orders_enriched
AT(OFFSET => -24*60*60);

-- Undrop table
UNDROP TABLE accidentally_dropped_table;

-- Create secure view with row-level security
CREATE OR REPLACE SECURE VIEW customer_orders_view AS
SELECT *
FROM orders_enriched
WHERE customer_id IN (
    SELECT customer_id FROM customer_access
    WHERE user_role = CURRENT_ROLE()
);""",
                domain="sql",
                subdomain="snowflake",
                tags=["variant", "time_travel", "flatten", "secure_view"],
                difficulty="advanced"
            ),
        ]

    def _sparksql_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Write SparkSQL queries for large-scale data processing",
                code="""-- SparkSQL: Large-scale analytics

-- Create managed table with partitioning
CREATE TABLE IF NOT EXISTS events (
    event_id BIGINT,
    event_type STRING,
    user_id BIGINT,
    properties MAP<STRING, STRING>,
    event_timestamp TIMESTAMP
)
USING DELTA
PARTITIONED BY (event_date DATE)
CLUSTERED BY (user_id) INTO 256 BUCKETS
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- Window functions for sessionization
WITH events_with_session AS (
    SELECT
        user_id,
        event_timestamp,
        event_type,
        -- Detect session boundaries (30 min gap)
        CASE
            WHEN LAG(event_timestamp) OVER (PARTITION BY user_id ORDER BY event_timestamp) IS NULL
                OR unix_timestamp(event_timestamp) - unix_timestamp(
                    LAG(event_timestamp) OVER (PARTITION BY user_id ORDER BY event_timestamp)
                ) > 1800
            THEN 1
            ELSE 0
        END as is_session_start
    FROM events
    WHERE event_date = current_date()
),
sessions AS (
    SELECT
        user_id,
        event_timestamp,
        event_type,
        SUM(is_session_start) OVER (
            PARTITION BY user_id
            ORDER BY event_timestamp
            ROWS UNBOUNDED PRECEDING
        ) as session_id
    FROM events_with_session
)
SELECT
    user_id,
    session_id,
    MIN(event_timestamp) as session_start,
    MAX(event_timestamp) as session_end,
    COUNT(*) as events_in_session,
    COLLECT_LIST(event_type) as event_sequence
FROM sessions
GROUP BY user_id, session_id;

-- Explode nested structures
SELECT
    user_id,
    event_date,
    property_key,
    property_value
FROM events
LATERAL VIEW EXPLODE(properties) props AS property_key, property_value
WHERE property_key IN ('source', 'campaign', 'medium');

-- Pivot table for metrics
SELECT *
FROM (
    SELECT event_date, event_type, COUNT(*) as cnt
    FROM events
    WHERE event_date >= current_date() - INTERVAL 7 DAYS
    GROUP BY event_date, event_type
)
PIVOT (
    SUM(cnt)
    FOR event_type IN ('page_view', 'click', 'purchase', 'signup')
)
ORDER BY event_date;""",
                domain="sql",
                subdomain="sparksql",
                tags=["delta", "window_functions", "sessionization"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Write SparkSQL with Delta Lake operations",
                code="""-- SparkSQL: Delta Lake operations

-- Create Delta table with schema evolution
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id BIGINT,
    email STRING,
    name STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
USING DELTA
LOCATION '/datalake/user_profiles'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.columnMapping.mode' = 'name',
    'delta.minReaderVersion' = '2',
    'delta.minWriterVersion' = '5'
);

-- Merge (upsert) operation
MERGE INTO user_profiles AS target
USING (
    SELECT
        user_id,
        email,
        name,
        COALESCE(created_at, current_timestamp()) as created_at,
        current_timestamp() as updated_at
    FROM staging_users
) AS source
ON target.user_id = source.user_id
WHEN MATCHED THEN UPDATE SET
    email = source.email,
    name = source.name,
    updated_at = source.updated_at
WHEN NOT MATCHED THEN INSERT *;

-- Delete with history
DELETE FROM user_profiles
WHERE email LIKE '%@deleted.com';

-- Time travel
SELECT * FROM user_profiles VERSION AS OF 10;
SELECT * FROM user_profiles TIMESTAMP AS OF '2024-01-15 10:00:00';

-- Describe history
DESCRIBE HISTORY user_profiles LIMIT 10;

-- Restore to previous version
RESTORE TABLE user_profiles TO VERSION AS OF 5;

-- Optimize table
OPTIMIZE user_profiles
ZORDER BY (user_id);

-- Vacuum old files (retain 7 days)
VACUUM user_profiles RETAIN 168 HOURS;

-- Read change data feed
SELECT *
FROM table_changes('user_profiles', 5, 10)
WHERE _change_type IN ('insert', 'update_postimage');

-- Clone table
CREATE TABLE user_profiles_backup
SHALLOW CLONE user_profiles
VERSION AS OF 10;""",
                domain="sql",
                subdomain="sparksql",
                tags=["delta_lake", "merge", "time_travel", "cdf"],
                difficulty="advanced"
            ),
        ]

    def _common_patterns(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Write SQL for slowly changing dimension Type 2",
                code="""-- SCD Type 2: Track historical changes
-- Works across most SQL databases with minor syntax adjustments

-- Dimension table structure
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,  -- Surrogate key
    customer_id INT NOT NULL,      -- Business key
    name VARCHAR(100),
    email VARCHAR(255),
    address VARCHAR(500),
    tier VARCHAR(20),
    -- SCD Type 2 columns
    effective_date DATE NOT NULL,
    expiration_date DATE,
    is_current BOOLEAN DEFAULT TRUE,
    -- Audit columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_customer_business_key ON dim_customer(customer_id, is_current);
CREATE INDEX idx_customer_effective ON dim_customer(effective_date, expiration_date);

-- Merge procedure for SCD Type 2
-- Step 1: Expire changed records
UPDATE dim_customer d
SET
    expiration_date = CURRENT_DATE - INTERVAL '1 day',
    is_current = FALSE,
    updated_at = CURRENT_TIMESTAMP
FROM staging_customers s
WHERE d.customer_id = s.customer_id
    AND d.is_current = TRUE
    AND (
        d.name != s.name OR
        d.email != s.email OR
        d.address != s.address OR
        d.tier != s.tier
    );

-- Step 2: Insert new versions for changed records
INSERT INTO dim_customer (
    customer_key, customer_id, name, email, address, tier,
    effective_date, expiration_date, is_current
)
SELECT
    NEXTVAL('customer_key_seq'),
    s.customer_id,
    s.name,
    s.email,
    s.address,
    s.tier,
    CURRENT_DATE,
    NULL,
    TRUE
FROM staging_customers s
LEFT JOIN dim_customer d ON s.customer_id = d.customer_id AND d.is_current = TRUE
WHERE d.customer_id IS NULL  -- New customers
   OR (  -- Changed customers (already expired above)
        d.name != s.name OR
        d.email != s.email OR
        d.address != s.address OR
        d.tier != s.tier
    );

-- Query point-in-time state
SELECT *
FROM dim_customer
WHERE customer_id = 12345
    AND '2024-06-15' BETWEEN effective_date
        AND COALESCE(expiration_date, '9999-12-31');""",
                domain="sql",
                subdomain="analytics",
                tags=["scd", "data_warehouse", "dimension"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Write efficient SQL pagination with keyset pagination",
                code="""-- Keyset pagination: Much faster than OFFSET for large tables
-- Works across PostgreSQL, MySQL, SQLite, etc.

-- BAD: Offset pagination (slow for large offsets)
SELECT * FROM products
ORDER BY created_at DESC, id DESC
LIMIT 20 OFFSET 100000;  -- Has to scan 100,000 rows!

-- GOOD: Keyset pagination (consistent performance)
-- First page
SELECT id, name, price, created_at
FROM products
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Next page: use last row's values as cursor
SELECT id, name, price, created_at
FROM products
WHERE (created_at, id) < ('2024-01-15 10:30:00', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- For bidirectional pagination
-- Encode cursor as base64 JSON
-- cursor = base64('{"created_at":"2024-01-15T10:30:00","id":12345}')

-- Previous page (reverse sort, then reverse results)
SELECT * FROM (
    SELECT id, name, price, created_at
    FROM products
    WHERE (created_at, id) > ('2024-01-15 10:30:00', 12345)
    ORDER BY created_at ASC, id ASC
    LIMIT 20
) subq
ORDER BY created_at DESC, id DESC;

-- With additional filters
SELECT id, name, price, created_at
FROM products
WHERE category_id = 5
    AND is_active = TRUE
    AND (created_at, id) < ('2024-01-15 10:30:00', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Required index for performance
CREATE INDEX idx_products_pagination
ON products (created_at DESC, id DESC)
WHERE is_active = TRUE;""",
                domain="sql",
                subdomain="optimization",
                tags=["pagination", "performance", "keyset"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Write SQL for calculating running totals and cumulative metrics",
                code="""-- Running totals and cumulative metrics
-- Compatible with PostgreSQL, MySQL 8+, SQLite 3.25+, Redshift, Snowflake

-- Basic running total
SELECT
    transaction_date,
    amount,
    SUM(amount) OVER (
        ORDER BY transaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM transactions
ORDER BY transaction_date;

-- Running total by category
SELECT
    category,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY category
        ORDER BY transaction_date
        ROWS UNBOUNDED PRECEDING
    ) AS category_running_total
FROM transactions
ORDER BY category, transaction_date;

-- Cumulative distinct count
SELECT
    order_date,
    COUNT(DISTINCT customer_id) AS daily_customers,
    SUM(COUNT(DISTINCT customer_id)) OVER (
        ORDER BY order_date
    ) AS cumulative_customers_approx,
    -- Exact cumulative distinct (subquery approach)
    (SELECT COUNT(DISTINCT customer_id)
     FROM orders o2
     WHERE o2.order_date <= o1.order_date) AS cumulative_customers_exact
FROM orders o1
GROUP BY order_date
ORDER BY order_date;

-- YTD (Year-to-Date) calculations
SELECT
    DATE_TRUNC('month', order_date) AS month,
    SUM(total_amount) AS monthly_revenue,
    SUM(SUM(total_amount)) OVER (
        PARTITION BY EXTRACT(YEAR FROM order_date)
        ORDER BY DATE_TRUNC('month', order_date)
    ) AS ytd_revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date), EXTRACT(YEAR FROM order_date)
ORDER BY month;

-- Moving averages (7-day, 30-day)
SELECT
    sale_date,
    daily_revenue,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_7d,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_30d
FROM (
    SELECT
        DATE(order_date) AS sale_date,
        SUM(total_amount) AS daily_revenue
    FROM orders
    GROUP BY DATE(order_date)
) daily_sales
ORDER BY sale_date;""",
                domain="sql",
                subdomain="analytics",
                tags=["window_functions", "running_total", "analytics"],
                difficulty="intermediate"
            ),
        ]
