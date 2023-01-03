CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.np_acl (
    id BIGINT,
    year INT,
    citations ARRAY<BIGINT>,
    noun_chunks ARRAY<STRING>
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-s2-lucas/s2orc_20221211/acl_np_cits_ascii_clean/np'

----------------

UNLOAD (
    -- SUBTABLE 1: only papers that have been cited at least n times
    --             in the corpus will be included in the forecite analysis
    --             Its columns are:
    --                 id: the id of the paper p
    --                 year: the year paper p was published
    --                 noun_chunks: the noun chunks in p
    WITH cited_subset AS (
        -- Given a paper id, we get its id, when it was published,
        -- and the noun chunks in the paper.
        SELECT
            og.id,
            og.year,
            og.noun_chunks
        FROM "temp_lucas"."np_acl" AS og
        INNER JOIN (
            -- we first "invert" the citations to go from paper -> cited papers
            -- to paper -> citing paper.
            SELECT cited
            FROM (
                SELECT
                    cited,
                    id
                FROM "temp_lucas"."np_acl"
                CROSS JOIN UNNEST(citations) AS t(cited)
            )
            GROUP BY cited
            -- PARAMETER 1: minimum number of times a paper must be cited
            --              to be included in the forecite analysis
            HAVING COUNT(id) >= 3
        ) AS cs
            ON og.id = cs.cited
    ),
    -- SUBTABLE 2: this table contains all noun chunks that appear in
    --             the corpus at least n times.
    --             Its columns are:
    --                 term: the term appearing in the corpus at least n times
    terms_subset AS (
        SELECT term
        FROM "temp_lucas"."np_acl"
        CROSS JOIN UNNEST(noun_chunks) AS t(term)
        GROUP BY term
        -- PARAMETER 2: minimum number of times a term must appear in
        --              the corpus to be included in the forecite analysis
        HAVING COUNT(id) >= 3
    ),
    -- SUBTABLE 3: this table contains all terms that appear in any
    --             paper that could be the defining paper for the term.
    --             Its columns are:
    --                 cited: the id of the paper p that could contain the
    --                        definition of the term
    --                 cited_year: the year paper p was published
    --                 noun_chunks_map: a map from noun chunks to the number
    --                                  in papers in the corpus that cite p
    cited_terms AS (
        SELECT
            cited,
            cited_year,
            -- the nested operations here transform an array of arrays of
            -- noun chunks (oputput of ARRAY_AGG) into a map from noun chunks
            -- to the number of times the noun chunk appears in the nested
            -- arrays.
            TRANSFORM_VALUES(
                -- from table to map
                MULTIMAP_FROM_ENTRIES(
                    -- from list to table
                    TRANSFORM(
                        -- from list of lists to list
                        FLATTEN(ARRAY_AGG(shared_noun_chunks)),
                        x -> ROW(x, 1)
                    )
                ),
                -- merge counts
                (k, v) -> REDUCE(v, 0, (s, x) -> s + x, s -> s)
            ) AS noun_chunks_map
        FROM (
            SELECT
                og.id AS citing,
                cs.id AS cited,
                cs.year AS cited_year,
                -- we intersect the noun chunks in the citing paper with
                -- the noun chunks in the cited paper. We only care about
                -- terms that appear in both!
                ARRAY_INTERSECT(
                    og.noun_chunks,
                    cs.noun_chunks
                ) AS shared_noun_chunks

            FROM "temp_lucas"."np_acl" AS og
            CROSS JOIN UNNEST(citations) AS t(cited)
            INNER JOIN cited_subset AS cs
                ON t.cited = cs.id
        )
        GROUP by cited, cited_year
    ),
    year_terms AS (
        SELECT
            t.term,
            -- HISTOGRAM goes from a list of year (repeating)
            -- to a (year, count) map.
            HISTOGRAM(year) AS counts
        FROM "temp_lucas"."np_acl"
        CROSS JOIN UNNEST(noun_chunks) AS t(term)
        INNER JOIN (
            SELECT DISTINCT term
            FROM cited_terms
            CROSS JOIN UNNEST(noun_chunks_map) AS t(term, counts)
        ) AS tc
            ON t.term = tc.term
        GROUP BY t.term
    ),
    forecite_stats AS (
    SELECT
        t.term,
        ct.cited,
        ct.cited_year AS year,
        t.counts AS term_count,
        ct.cited_year,
        -- this set of nested operations goes from a map from year to count
        -- to the sum of counts for years greater than or equal to the year
        -- the paper was published.
        REDUCE(
            -- get only the values from the map
            MAP_VALUES(
                -- filter out years before the paper was published
                MAP_FILTER(yt.counts, (k, v) -> (k >= ct.cited_year))
            ),
            -- this reduce sums the values; AWS Athena does not support
            -- ARRAY_SUM, so we have to do it manually.
            0, (s, x) -> s + x, s -> s
        ) AS year_count
    FROM cited_terms AS ct
    CROSS JOIN UNNEST(noun_chunks_map) AS t(term, counts)
    INNER JOIN year_terms AS yt
        ON t.term = yt.term
    -- PARAMETER 3: minimum number of times a term must be shared between
    --              cited paper and its citing papers to be included in
    --              the forecite analysis.
    WHERE t.counts >= 3
)
SELECT
    -- make sure terms do not contain tabs or newlines
    REPLACE(REPLACE(term, chr(9), ' '), chr(10), ' ') AS term,
    cited,
    year,
    term_count,
    year_count,
    -- this is the main forecite equation
    (
        LOG2(CAST(term_count AS REAL) + 1.0) *
        (CAST(term_count AS REAL) / CAST(year_count AS REAL))
    ) AS forecite_score
FROM forecite_stats
-- this is a bug in Athena? counts are weird sometimes!
-- but intuitively the year count must be greater than the term count,
-- because the term count is the number of future papers that cite the
-- target paper and contain the term, while the year count is just the
-- number of future papers that contain the term.
WHERE term_count <= year_count
ORDER BY forecite_score DESC
)
TO 's3://ai2-s2-lucas/s2orc_20221211/acl_forecite/'
WITH (
    format='TEXTFILE',
    field_delimiter = '\t'
)
