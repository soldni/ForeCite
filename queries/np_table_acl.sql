CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.np_acl (
    id BIGINT,
    year INT,
    citations ARRAY<BIGINT>,
    noun_chunks ARRAY<STRING>
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-s2-lucas/s2orc_20221211/acl_np_cits_ascii/np'

----------------


UNLOAD (
    SELECT
        -- make sure no tabs or newlines
        REPLACE(REPLACE(term, chr(9), ' '), chr(10), ' ') AS term,
        cited,
        year,
        term_count,
        year_count,
        (
            LOG2(CAST(term_count AS REAL) + 1.0) *
            (CAST(term_count AS REAL) / CAST(year_count AS REAL))
        ) AS forecite_score
    FROM (
        SELECT
            pt.term,
            pt.cited,
            pt.year,
            pt.term_count,
            SUM(yt.year_count) as year_count
        FROM (
            SELECT
                tt.cited,
                tt.term,
                tt.term_count,
                og.year
            FROM (
                SELECT
                    cited,
                    term,
                    COUNT(id) AS term_count
                FROM (
                    SELECT
                        cited,
                        id,
                        term
                    FROM (
                        SELECT
                            cited,
                            id,
                            noun_chunks
                        FROM "temp_lucas"."np_acl"
                        CROSS JOIN UNNEST(citations) as t(cited)
                    )
                    CROSS JOIN UNNEST(noun_chunks) as t(term)
                )
                GROUP BY cited, term
            ) AS tt
            INNER JOIN "temp_lucas"."np_acl" AS og
                ON og.id = tt.cited
            WHERE  CONTAINS(og.noun_chunks, tt.term)
        ) AS pt
        INNER JOIN (
            SELECT
                term,
                year,
                COUNT(id) as year_count
            FROM (
                SELECT
                    term,
                    year,
                    id
                FROM "temp_lucas"."np_acl"
                CROSS JOIN UNNEST(noun_chunks) as t(term)
            )
            GROUP BY term, year
        ) AS yt
            ON yt.term = pt.term AND yt.year >= pt.year
        GROUP BY
            pt.term,
            pt.cited,
            pt.year,
            pt.term_count
    )
    -- this is a bug in Athena? counts are weird sometimes!
    -- but intuitively the year count must be greater than the term count,
    -- because the term count is the number of future papers that cite the
    -- target paper and contain the term, while the year count is just the
    -- number of future papers that contain the term.
    WHERE year_count >= term_count
    ORDER BY forecite_score DESC, term
)
TO 's3://ai2-s2-lucas/s2orc_20221211/acl_forecite/'
WITH (
    format='TEXTFILE',
    field_delimiter = '\t'
)
