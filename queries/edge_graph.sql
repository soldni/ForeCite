-- This is an Amazon Athena query to be executed in the s2orc_papers database.
-- It is used to generate the citation graph for the S2ORC dataset.
--
-- The output is a table with two columns: citing_id and cited_id.
-- The citing_id is the id of the paper that cites the cited_id in the
-- semantic scholar corpus.

UNLOAD (
    SELECT
        id,
        year,
        ARRAY_SORT(ARRAY_AGG(CAST(cited AS bigint))) AS cited,
        floor(rand() * 30) as part_id
    FROM (
        SELECT
            id,
            year,
            JSON_EXTRACT(bib_entry, '$.attributes.matched_paper_id') as cited
        FROM (
            SELECT
                id,
                metadata.publication_date.year as year,
                cast(
                    json_parse(content.grobid.annotations.bib_entry) AS array(json)
                ) as bib_entries
            FROM s2orc_papers.releases
            WHERE year=2022 AND month=12 AND day=11
        )
        CROSS JOIN UNNEST(bib_entries) as t(bib_entry)
    )
    WHERE cited is not NULL
    GROUP BY id, year
)
TO 's3://ai2-s2-lucas/s2orc_20221211/edge_graph_sorted/'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['part_id']
)
