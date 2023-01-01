UNLOAD (
    WITH acl_papers AS (
            SELECT corpus_paper_id AS id
            FROM content_ext.paper_sources
            WHERE source = 'ACL'
        ),
    s2orc_subset AS (
        SELECT
            s2orc.*
        FROM (
            SELECT
                id,
                metadata.publication_date.year AS year,
                metadata.title AS title,
                metadata.abstract AS abstract,
                CAST (
                    JSON_PARSE(metadata.fields_of_study) AS ARRAY(varchar)
                ) AS fields_of_study,
                content.grobid.contents AS full_text,
                (
                    CAST(
                        JSON_PARSE(content.grobid.annotations.paragraph)
                        AS ARRAY(json)
                    ) ||
                    CAST(
                        JSON_PARSE(content.grobid.annotations.section_header)
                        AS ARRAY(json)
                    )
                ) AS all_paralocs,
                (
                    CAST(
                        IF (
                            content.grobid.annotations.figure_ref
                            IS NOT NULL,
                            JSON_PARSE(
                                content.grobid.annotations.figure_ref
                            ),
                            JSON_PARSE('[]')
                        )
                        AS ARRAY(json)
                    ) ||
                    CAST(
                        IF (
                            content.grobid.annotations.bib_ref
                            IS NOT NULL,
                            JSON_PARSE(
                                content.grobid.annotations.bib_ref
                            ),
                            JSON_PARSE('[]')
                        )
                        AS ARRAY(json)
                    ) ||
                    CAST(
                        IF (
                            content.grobid.annotations.table_ref
                            IS NOT NULL,
                            JSON_PARSE(
                                content.grobid.annotations.table_ref
                            ),
                            JSON_PARSE('[]')
                        )
                        AS ARRAY(json)
                    )
                ) AS all_internal_refs
            FROM s2orc_papers.releases
            WHERE
                year=2022 AND
                month=12 AND
                day=11
        ) AS s2orc
        INNER JOIN acl_papers AS acl
            ON s2orc.id = acl.id
    ),
    citations_table AS (
        SELECT
            cits.citing_corpus_paperid AS id,
            ARRAY_AGG(cits.cited_corpus_paperid) AS citations
        FROM "content_ext"."citations"  AS cits
        INNER JOIN (
            SELECT id
            FROM s2orc_subset
        ) AS subs
            ON cits.citing_corpus_paperid = subs.id
        GROUP BY cits.citing_corpus_paperid
    ),
    prepared_locs AS (
        SELECT
            id,
            year,
            title,
            abstract,
            full_text,
            ARRAY_JOIN(fields_of_study, ',') AS fields_of_study,
            ARRAY_SORT(
                ARRAY_DISTINCT(
                    TRANSFORM(
                        all_paralocs,
                        x -> (
                            ARRAY [CAST(JSON_EXTRACT(x, '$.start') AS INT)] ||
                            ARRAY [CAST(JSON_EXTRACT(x, '$.end') AS INT)]
                        )
                    )
                )
            ) AS all_paralocs,
            ARRAY_DISTINCT(
                TRANSFORM(
                    all_internal_refs,
                    x -> (
                        ARRAY [CAST(JSON_EXTRACT(x, '$.start') AS INT)] ||
                        ARRAY [CAST(JSON_EXTRACT(x, '$.end') AS INT)]
                    )
                )
            ) AS all_internal_refs
        FROM s2orc_subset
    ),
    cleaned_text_table AS (
        SELECT
            id,
            year,
            title,
            abstract,
            fields_of_study,
            all_paralocs,
            REDUCE(
                all_internal_refs,
                full_text,
                (part_clean, loc) -> (
                    -- everying before this location, concatenated with...
                    SUBSTR(part_clean, 1, loc[1]) ||
                    -- ...a series of "␂/␃" symbols of the same length of the
                    -- reference, concatenated with...
                    LPAD('␃', loc[2] - loc[1], '␂') ||
                    -- ...the remaining of the partially cleaneds text
                    -- after the location.
                    SUBSTR(
                        part_clean,
                        loc[2] + 1,
                        LENGTH(part_clean) - loc[2] + 1
                    )
                ),
                s -> s
            ) AS cleaned_text
        FROM prepared_locs
    ),
    just_para_from_clean_text_table AS (
        SELECT
            id,
            year,
            title,
            abstract,
            fields_of_study,
            REGEXP_REPLACE(
                ARRAY_JOIN(
                    TRANSFORM(
                        all_paralocs,
                        paralocs -> NORMALIZE(
                            SUBSTR(
                                cleaned_text,
                                paralocs[1],
                                paralocs[2] - paralocs[1] + 1
                            )
                        )
                    ),
                -- add two unix newlines "\n" between paragraphs
                chr(10) || chr(10)
                ),
                -- btw, using the very uncommon start of text and end of text
                -- unicode symbols to make sure they don't clash with any other
                -- symbol in the text
                '␂*␃',
                ''
            ) AS full_text
            FROM cleaned_text_table
    )

    SELECT
        tt.*,
        ct.citations,
        -- make 30 partitions for smaller output files
        CAST(FLOOR(RAND() * 3) AS INT) AS part_id
    FROM just_para_from_clean_text_table AS tt
    INNER JOIN citations_table AS ct
        ON tt.id = ct.id
)
TO 's3://ai2-s2-lucas/s2orc_20221211/acl_content_cits_clean/'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['part_id']
)
